import numpy as np
import torch
import trimesh
import warp as wp
import mujoco

from typing import Optional, List, Union
from jaxtyping import Float, Bool, Int
from .helpers import quat_rotate_inverse, trimesh2wp
from .kernels import (
    raycast_kernel,
    raycast_against_meshes_kernel,
    transform_and_raycast_kernel,
    transform_and_raycast_against_meshes_kernel,
)

MeshType = Union[wp.Mesh, trimesh.Trimesh]


class MultiMeshRaycaster:
    """
    Raycaster that supports multiple and dynamic meshes.

    Args:
        meshes: List of wp.Mesh objects.
    """

    def __init__(self, meshes_wp: List[MeshType], device: str | torch.device):
        self.meshes_wp = [
            mesh_wp if isinstance(mesh_wp, wp.Mesh) else trimesh2wp(mesh_wp, device)
            for mesh_wp in meshes_wp
        ]
        if not isinstance(device, str):
            device = wp.get_device(str(device))
        self.device = device
        self.mesh_names = None
        self.initialized = False
    
    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.meshes_array = wp.array([mesh_wp.id for mesh_wp in self.meshes_wp], device=self.device, dtype=wp.uint64)
    
    def add_mesh(self, mesh: MeshType):
        """
        Add a mesh to the raycaster.

        Args:
            mesh: The mesh to add. Can be a trimesh.Trimesh object or a wp.Mesh object.
        """
        if isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh2wp(mesh, self.device)
        self.meshes_wp.append(mesh)
        self.initialized = False
    
    def add_from_path(
        self,
        path: str | list[str],
        stage: "Usd.Stage",
        combine: bool = True,
    ) -> List[str]:
        """
        Add meshes from USD prim paths to the raycaster.

        Args:
            path: Prim path(s) to search for (supports regex). Can be a single string or list of strings.
            stage: The USD stage to search in.
            combine: If True, combine all matching meshes into one. If False, add each mesh separately.
        returns:
            List of prim paths that were found.
        """
        
        if isinstance(path, str):
            path = [path]
        path_expr = "(" + "|".join(path) + ")"
        from .utils_usd import find_matching_prims, get_trimesh_from_prim
        prims = find_matching_prims(path_expr, stage)
        trimesh_list = []
        
        for prim in prims:
            mesh = get_trimesh_from_prim(prim)
            trimesh_list.append(mesh)
        
        if combine:
            mesh = trimesh.util.concatenate(trimesh_list)
            self.add_mesh(mesh)
        else:
            for mesh in trimesh_list:
                self.add_mesh(mesh)
        self.initialized = False
        return [prim.GetPath().pathString for prim in prims]

    @property
    def n_points(self):
        return sum(mesh_wp.points.shape[0] for mesh_wp in self.meshes_wp)

    @property
    def n_faces(self):
        return sum(mesh_wp.indices.reshape((-1, 3)).shape[0] for mesh_wp in self.meshes_wp)

    @property
    def n_meshes(self):
        return len(self.meshes_wp)

    def __repr__(self) -> str:
        return f"MultiMeshRaycaster(n_meshes={self.n_meshes}, n_points={self.n_points}, n_faces={self.n_faces})"

    def raycast(
        self,
        mesh_pos_w: Float[torch.Tensor, "N n_meshes 3"],  # [N, n_meshes, 3]
        mesh_quat_w: Float[torch.Tensor, "N n_meshes 4"],  # [N, n_meshes, 4]
        ray_starts_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        ray_dirs_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
        *,
        enabled: Optional[Bool[torch.Tensor, "N"]]=None,  # [N]
        mesh_indices: Optional[Int[torch.Tensor, "N n_meshes"]]=None,  # [N, n_meshes]
    ) -> tuple[Float[torch.Tensor, "N n_rays 3"], Float[torch.Tensor, "N n_rays"]]:
        """
        Perform raycasting against multiple meshes.
        
        For each batch element, rays are cast against all meshes (or a subset specified
        by `mesh_indices`), and the closest hit point across all meshes is returned for
        each ray. Rays that don't hit any mesh within the distance range will have
        a hit distance equal to `max_dist`.
        
        Args:
            mesh_pos_w: The position of each mesh in the world frame. Shape [N, n_meshes, 3],
                where N is the batch size and n_meshes is the number of meshes. Each row
                represents the 3D position (x, y, z) of a mesh in world coordinates.
            mesh_quat_w: The orientation of each mesh in the world frame as quaternions.
                Shape [N, n_meshes, 4]. Quaternions are expected in WXYZ format (scalar first).
                Must be normalized unit quaternions.
            ray_starts_w: The starting points of the rays in the world frame. Shape [N, n_rays, 3],
                where n_rays is the number of rays per batch element. Each row represents
                the 3D starting position (x, y, z) of a ray.
            ray_dirs_w: The direction vectors of the rays in the world frame. Shape [N, n_rays, 3].
                Direction vectors should be normalized (unit vectors). Each row represents
                the 3D direction vector of a ray.
            min_dist: The minimum distance along the ray to consider for hits. Rays that hit
                closer than this distance will be ignored. Defaults to 0.0. Useful for avoiding
                self-intersections or ignoring hits at the ray origin.
            max_dist: The maximum distance along the ray to search for hits. Rays that don't
                hit within this distance will return `max_dist` as the hit distance. Defaults
                to 100.0. Acts as the maximum ray length.
            enabled: Optional boolean tensor indicating which batch elements should be processed.
                Shape [N]. If None (default), all batch elements are enabled. Disabled elements
                will have their hit distances set to infinity. This is a keyword-only argument.
            mesh_indices: Optional tensor specifying which meshes to raycast against for each
                batch element. Shape [N, n_meshes], where each element is an index into the
                mesh array. If None (default), all meshes are tested. When provided, allows
                different batch elements to test against different subsets of meshes. This is
                a keyword-only argument.
        
        Returns:
            tuple: A tuple containing:
                - hit_positions (torch.Tensor): The 3D positions of the closest hit points
                  in the world frame. Shape [N, n_rays, 3]. Computed as `ray_starts_w + 
                  hit_distances * ray_dirs_w`. For rays that don't hit, positions correspond
                  to the point at `max_dist` along the ray direction.
                - hit_distances (torch.Tensor): The distances along each ray to the closest
                  hit point. Shape [N, n_rays]. Values are in the range [min_dist, max_dist],
                  or `max_dist` if no hit occurred. The minimum is taken across all meshes
                  for each ray.
        
        Note:
            - This method automatically initializes the raycaster if not already initialized.
            - The coordinate transformation (world to mesh-local) is performed on the CPU
              before calling the GPU kernel, unlike `raycast_fused()` which performs the
              transformation on the GPU.
            - When `mesh_indices` is provided, the second dimension of `mesh_pos_w` and
              `mesh_quat_w` should match `mesh_indices.shape[1]`, not necessarily `n_meshes`.
            - All input tensors must be on the same device as the raycaster.
        """
        if not self.initialized:
            self.initialize()
        
        n_rays = ray_dirs_w.shape[1]
        N = mesh_pos_w.shape[0]

        if enabled is None:
            enabled = torch.ones(N, dtype=torch.bool, device=ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        if mesh_indices is None:
            result_shape = (N, self.n_meshes, n_rays)
            mesh_pos_w = mesh_pos_w.reshape(N, self.n_meshes, 1, 3)  # [N, n_meshes, 1, 3]
            mesh_quat_w = mesh_quat_w.reshape(N, self.n_meshes, 1, 4)  # [N, n_meshes, 1, 4]

            # convert to mesh frame
            ray_starts_b = quat_rotate_inverse(mesh_quat_w, ray_starts_w.unsqueeze(1) - mesh_pos_w)
            ray_dirs_b = quat_rotate_inverse(mesh_quat_w, ray_dirs_w.unsqueeze(1))

            ray_starts_wp = wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True)
            ray_dirs_wp = wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True)
            enabled_wp = wp.from_torch(enabled, dtype=wp.bool, return_ctype=True)

            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            wp.launch(
                raycast_kernel,
                dim=(N, self.n_meshes, n_rays),
                inputs=[
                    self.meshes_array,
                    ray_starts_wp,
                    ray_dirs_wp,
                    enabled_wp,
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                "`mesh_indices` must have the same number of meshes as `mesh_pos_w` and `mesh_quat_w`"
            )
            n_meshes = mesh_indices.shape[1]
            result_shape = (N, n_meshes, n_rays)
            mesh_pos_w = mesh_pos_w.reshape(N, n_meshes, 1, 3)  # [N, n_meshes, 1, 3]
            mesh_quat_w = mesh_quat_w.reshape(N, n_meshes, 1, 4)  # [N, n_meshes, 1, 4]

            # convert to mesh frame
            ray_starts_b = quat_rotate_inverse(mesh_quat_w, ray_starts_w.unsqueeze(1) - mesh_pos_w)
            ray_dirs_b = quat_rotate_inverse(mesh_quat_w, ray_dirs_w.unsqueeze(1))

            ray_starts_wp = wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True)
            ray_dirs_wp = wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True)
            enabled_wp = wp.from_torch(enabled, dtype=wp.bool, return_ctype=True)
            mesh_indices_wp = wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True)

            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            wp.launch(
                raycast_against_meshes_kernel,
                dim=(N, n_meshes, n_rays),
                inputs=[
                    self.meshes_array,
                    mesh_indices_wp,
                    ray_starts_wp,
                    ray_dirs_wp,
                    enabled_wp,
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )
        
        hit_distances = hit_distances.min(dim=1).values # [N, n_rays]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w # [N, n_rays, 3]
        return hit_positions, hit_distances
    
    def raycast_fused(
        self,
        mesh_pos_w: Float[torch.Tensor, "N n_meshes 3"],  # [N, n_meshes, 3]
        mesh_quat_w: Float[torch.Tensor, "N n_meshes 4"],  # [N, n_meshes, 4]
        ray_starts_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        ray_dirs_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
        *,
        enabled: Optional[Bool[torch.Tensor, "N"]]=None,  # [N]
        mesh_indices: Optional[Int[torch.Tensor, "N n_meshes"]]=None,  # [N, n_meshes]
    ) -> tuple[Float[torch.Tensor, "N n_rays 3"], Float[torch.Tensor, "N n_rays"]]:
        """
        Perform raycasting against multiple meshes using a fused GPU kernel.
        
        This method is an optimized version of `raycast()` that fuses the coordinate
        transformation and raycasting operations into a single GPU kernel, reducing
        memory transfers and improving performance. The transformation from world
        coordinates to mesh-local coordinates is performed on the GPU within the
        kernel, eliminating the need for intermediate tensor allocations.
        
        For each batch element, rays are cast against all meshes (or a subset specified
        by `mesh_indices`), and the closest hit point across all meshes is returned for
        each ray. Rays that don't hit any mesh within the distance range will have
        a hit distance equal to `max_dist`.
        
        Args:
            mesh_pos_w: The position of each mesh in the world frame. Shape [N, n_meshes, 3],
                where N is the batch size and n_meshes is the number of meshes. Each row
                represents the 3D position (x, y, z) of a mesh in world coordinates.
            mesh_quat_w: The orientation of each mesh in the world frame as quaternions.
                Shape [N, n_meshes, 4]. Quaternions are expected in WXYZ format (scalar first).
                Must be normalized unit quaternions.
            ray_starts_w: The starting points of the rays in the world frame. Shape [N, n_rays, 3],
                where n_rays is the number of rays per batch element. Each row represents
                the 3D starting position (x, y, z) of a ray.
            ray_dirs_w: The direction vectors of the rays in the world frame. Shape [N, n_rays, 3].
                Direction vectors should be normalized (unit vectors). Each row represents
                the 3D direction vector of a ray.
            min_dist: The minimum distance along the ray to consider for hits. Rays that hit
                closer than this distance will be ignored. Defaults to 0.0. Useful for avoiding
                self-intersections or ignoring hits at the ray origin.
            max_dist: The maximum distance along the ray to search for hits. Rays that don't
                hit within this distance will return `max_dist` as the hit distance. Defaults
                to 100.0. Acts as the maximum ray length.
            enabled: Optional boolean tensor indicating which batch elements should be processed.
                Shape [N]. If None (default), all batch elements are enabled. Disabled elements
                will have their hit distances set to infinity. This is a keyword-only argument.
            mesh_indices: Optional tensor specifying which meshes to raycast against for each
                batch element. Shape [N, n_meshes], where each element is an index into the
                mesh array. If None (default), all meshes are tested. When provided, allows
                different batch elements to test against different subsets of meshes. This is
                a keyword-only argument.
        
        Returns:
            tuple: A tuple containing:
                - hit_positions (torch.Tensor): The 3D positions of the closest hit points
                  in the world frame. Shape [N, n_rays, 3]. Computed as `ray_starts_w + 
                  hit_distances * ray_dirs_w`. For rays that don't hit, positions correspond
                  to the point at `max_dist` along the ray direction.
                - hit_distances (torch.Tensor): The distances along each ray to the closest
                  hit point. Shape [N, n_rays]. Values are in the range [min_dist, max_dist],
                  or `max_dist` if no hit occurred. The minimum is taken across all meshes
                  for each ray.
        
        Note:
            - This method automatically initializes the raycaster if not already initialized.
            - The coordinate transformation (world to mesh-local) is performed on the GPU
              within the kernel, making this more memory-efficient than `raycast()`.
            - When `mesh_indices` is provided, the second dimension of `mesh_pos_w` and
              `mesh_quat_w` should match `mesh_indices.shape[1]`, not necessarily `n_meshes`.
            - Quaternions are converted from WXYZ to XYZW format internally for Warp compatibility.
            - All input tensors must be on the same device as the raycaster.
        
        Example:
            >>> raycaster = MultiMeshRaycaster(meshes, device="cuda")
            >>> mesh_pos = torch.randn(10, 3, 3, device="cuda")  # 10 batches, 3 meshes
            >>> mesh_quat = torch.randn(10, 3, 4, device="cuda")
            >>> ray_starts = torch.randn(10, 100, 3, device="cuda")  # 100 rays per batch
            >>> ray_dirs = torch.randn(10, 100, 3, device="cuda")
            >>> ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)  # normalize
            >>> hit_pos, hit_dist = raycaster.raycast_fused(
            ...     mesh_pos, mesh_quat, ray_starts, ray_dirs,
            ...     min_dist=0.1, max_dist=50.0
            ... )
        """
        if not self.initialized:
            self.initialize()
        
        n_rays = ray_dirs_w.shape[1]
        N = mesh_pos_w.shape[0]
        assert mesh_pos_w.shape[1] == mesh_quat_w.shape[1] == self.n_meshes, (
            f"`mesh_pos_w` and `mesh_quat_w` must have the same number of meshes as the raycaster: {self.n_meshes}"
        )

        if enabled is None:
            enabled = torch.ones(N, dtype=torch.bool, device=ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        if mesh_indices is None:
            result_shape = (N, self.n_meshes, n_rays)
            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            wp.launch(
                transform_and_raycast_kernel,
                dim=(N, self.n_meshes, n_rays),
                inputs=[
                    self.meshes_array,
                    wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                "`mesh_indices` must have the same number of meshes as `mesh_pos_w` and `mesh_quat_w`"
            )
            n_meshes = mesh_indices.shape[1]
            result_shape = (N, n_meshes, n_rays)
            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            wp.launch(
                transform_and_raycast_against_meshes_kernel,
                dim=(N, mesh_indices.shape[1], n_rays),
                inputs=[
                    self.meshes_array,
                    wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                    wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )
        
        hit_distances = hit_distances.min(dim=1).values # [N, n_rays]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w # [N, n_rays, 3]
        return hit_positions, hit_distances

    @classmethod
    def from_prim_paths(
        cls,
        paths: list[str],
        stage: "Usd.Stage",
        device: str,
        simplify_factor: float = 0.0,
    ):
        """
        Args:
            paths: List of prim paths (can be regex) to find, e.g. ["World/.*/visuals"].
            stage: The USD stage to search in.
            device: The device to use for the raycaster.
            simplify_factor: The factor to simplify the meshes. 0.0 means no simplification.
                If a single float is provided, it will be used for all meshes.
        """
        if isinstance(simplify_factor, float):
            simplify_factor = [simplify_factor] * len(paths)
        if not len(paths) == len(simplify_factor):
            raise ValueError(
                "`simplify_factor` must be a single float or a list of floats with the same length as `paths`"
            )
        
        from .utils_usd import find_matching_prims, get_trimesh_from_prim

        meshes_wp = []
        n_verts_before = 0
        n_verts_after = 0
        n_faces_before = 0
        n_faces_after = 0
        for path, factor in zip(paths, simplify_factor):
            if not (prims := find_matching_prims(path, stage)):
                raise ValueError(f"No prims found for path {path}")

            for prim in prims:
                mesh_combined = get_trimesh_from_prim(prim)
                
                n_verts_before += mesh_combined.vertices.shape[0]
                n_faces_before += mesh_combined.faces.shape[0]
                if factor > 0.0:
                    mesh_combined = mesh_combined.simplify_quadric_decimation(factor)
                n_verts_after += mesh_combined.vertices.shape[0]
                n_faces_after += mesh_combined.faces.shape[0]

                meshes_wp.append(trimesh2wp(mesh_combined, device))

        if n_faces_before != n_faces_after:
            print(f"Simplified from ({n_verts_before}, {n_faces_before}) to ({n_verts_after}, {n_faces_after})")

        return cls(meshes_wp, device)
    

    @classmethod
    def from_MjModel(
        cls,
        body_names: list[str],
        model: mujoco.MjModel,
        device: str,
        simplify_factor: float = 0.0,
    ):
        """
        Args:
            model: The Mujoco model to use for the raycaster.
            device: The device to use for the raycaster.
            simplify_factor: The factor to simplify the meshes. 0.0 means no simplification.
                If a single float is provided, it will be used for all meshes.
        """
        from .utils_mjc import get_trimesh_from_body
        
        mesh_names = []
        meshes_wp = []

        n_verts_before = 0
        n_verts_after = 0
        n_faces_before = 0
        n_faces_after = 0

        for body_name in body_names:
            body = model.body(body_name)
            if body.geomnum.item() > 0:
                mesh = get_trimesh_from_body(body, model)
                n_verts_before += mesh.vertices.shape[0]
                n_faces_before += mesh.faces.shape[0]
                if simplify_factor > 0.0:
                    mesh = mesh.simplify_quadric_decimation(simplify_factor)
                n_verts_after += mesh.vertices.shape[0]
                n_faces_after += mesh.faces.shape[0]

                mesh_names.append(body.name)
                meshes_wp.append(trimesh2wp(mesh, device))

        if n_faces_before != n_faces_after:
            print(f"Simplified from ({n_verts_before}, {n_faces_before}) to ({n_verts_after}, {n_faces_after})")

        return cls(meshes_wp, device)

