"""
Benchmark script comparing different raycasting strategies and implementations.

This script benchmarks two different mesh organization strategies:
1. Option 1 (Independent Meshes): Each mesh is stored separately and positioned using
   mesh_pos_w and mesh_quat_w transformations. This allows selective raycasting against
   specific meshes using mesh_indices.
2. Option 2 (Combined Mesh): All meshes are concatenated into a single mesh, which is
   positioned at the origin. This requires fewer transformations but doesn't allow
   selective mesh testing.

For each strategy, the script compares two implementations:
- FUSED: Performs coordinate transformations on the GPU within the kernel (raycast_fused)
- NON-FUSED: Performs coordinate transformations on the CPU before calling the GPU kernel (raycast)

The script measures:
- Execution time per iteration (ms)
- Peak GPU memory usage
- Memory used during execution

Key Finding: Option 1 (independent meshes) is faster and uses less memory than Option 2
(combined mesh), as it allows more efficient GPU utilization and better memory locality
when only testing against specific meshes.

The script also verifies that all implementations produce identical results (within numerical
tolerance) to ensure correctness.
"""
import trimesh
import numpy as np
import torch
import random
import warp as wp
from tqdm import tqdm

from simple_raycaster.raycaster import MultiMeshRaycaster


def main():
    N, M = 4, 4
    # N, M = 16, 16
    # N, M = 64, 64
    spacing = 4.0
    # N_rays = 32 * 32
    N_rays = 4 * 4

    random_shapes = []
    random_names = []
    translations = []
    origin_x = -0.5 * (N - 1) * spacing
    origin_y = -0.5 * (M - 1) * spacing
    for i, j in tqdm(list(np.ndindex(N, M)), desc="Generating shapes"):
        shapes = []
        shape_A = trimesh.creation.box(extents=torch.rand(3) + torch.tensor([1.0, 1.0, 0.05]))
        shapes.append(shape_A)
        shape_B = trimesh.creation.box(extents=torch.rand(3) + torch.tensor([1.0, 1.0, 0.05]))
        shape_B.apply_translation(torch.randn(3).clamp(-3, 3) * 0.2)
        shapes.append(shape_B)
        shape_C = trimesh.creation.box(extents=torch.rand(3) + torch.tensor([1.0, 1.0, 0.05]))
        shape_C.apply_translation(torch.randn(3).clamp(-3, 3) * 0.2)
        shapes.append(shape_C)
        shape: trimesh.Trimesh = trimesh.util.concatenate(shapes)

        translation = [origin_x + i * spacing, origin_y + j * spacing, 0.0]
        
        translations.append(translation)
        random_shapes.append(shape)
        random_names.append(f"shape_{i}_{j}")
    
    shared_shape = trimesh.creation.box(extents=[N*spacing, M*spacing, 1.0])
    shared_shape.apply_translation([0., 0., -0.5])
    random_shapes.append(shared_shape)
    random_names.append("shared_shape")


    raycaster_independent = MultiMeshRaycaster(
        random_shapes,
        mesh_names=random_names,
        device="cuda",
    )
    mesh_filters = [[f"shape_{i}_{j}", "shared_shape"] for j in range(M) for i in range(N)]
    mesh_indices = raycaster_independent.get_mesh_ids(mesh_filters)
    
    import copy
    random_shapes = copy.deepcopy(random_shapes)
    for shape, translation in zip(random_shapes, translations):
        shape.apply_translation(translation)

    shapes_combined = trimesh.util.concatenate(random_shapes)
    raycaster_combined = MultiMeshRaycaster(
        [shapes_combined],
        device="cuda",
    )

    # if N * M <= 64:
    #     shapes_combined.show()

    # translations = torch.tensor(translations, device="cuda").reshape(N * M, 1, 3)
    # translations = torch.cat([
    #     translations,
    #     torch.zeros_like(translations), # for the shared shape
    # ], dim=1) # [N * M, 2, 3]
    # quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N * M, 1, 4).clone()
    # quaternions = torch.cat([
    #     quaternions,
    #     quaternions,
    # ], dim=1) # [N * M, 2, 4]
    translations = torch.tensor(translations, device="cuda").reshape(N * M, 3)
    translations_meshes = torch.cat([
        translations,
        torch.zeros_like(translations[:1]),
    ], dim=0) # [N * M + 1, 3]
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    quaternions = quaternions.expand(N * M + 1, 4).clone() # [N * M + 1, 4]
    
    ray_starts = translations[:, None, :].expand(-1, N_rays, -1).clone()
    ray_starts[:, :, :2] += torch.randn_like(ray_starts[:, :, :2]) * 0.05
    ray_starts[:, :, 2].uniform_(1.0, 2.0)

    ray_dirs = torch.zeros(N * M, N_rays, 3, device="cuda")
    ray_dirs[:, :, 2] = -1.0
    ray_dirs[:, :, :2].uniform_(-1.0, 1.0)
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    
    # Warmup runs
    print("Warming up...")
    for _ in range(10):
        # Fused versions
        _ = raycaster_independent.raycast_fused(
            mesh_pos_w=translations_meshes,
            mesh_quat_w=quaternions,
            ray_starts_w=ray_starts.clone(),
            ray_dirs_w=ray_dirs.clone(),
            mesh_indices=mesh_indices,
            min_dist=0.05,
            max_dist=10.0,
        )
        _ = raycaster_combined.raycast_fused(
            mesh_pos_w=torch.tensor([0., 0., 0.], device="cuda").expand(N * M + 1, 3),
            mesh_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N * M + 1, 4),
            ray_starts_w=ray_starts.clone(),
            ray_dirs_w=ray_dirs.clone(),
            min_dist=0.05,
            max_dist=10.0,
        )
        # # Non-fused versions
        # _ = raycaster_independent.raycast(
        #     mesh_pos_w=translations,
        #     mesh_quat_w=quaternions,
        #     ray_starts_w=ray_starts.clone(),
        #     ray_dirs_w=ray_dirs.clone(),
        #     mesh_indices=mesh_indices,
        #     min_dist=0.05,
        #     max_dist=10.0,
        # )
        # _ = raycaster_combined.raycast(
        #     mesh_pos_w=torch.tensor([0., 0., 0.], device="cuda").expand(N * M, 1, 3),
        #     mesh_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N * M, 1, 4),
        #     ray_starts_w=ray_starts.clone(),
        #     ray_dirs_w=ray_dirs.clone(),
        #     min_dist=0.05,
        #     max_dist=10.0,
        # )
    torch.cuda.synchronize()
    
    # Helper function to format memory size
    def format_memory(bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"
    
    num_iterations = 100
    
    # Helper function to benchmark a raycast call
    def benchmark_raycast(raycaster, method_name, mesh_pos_w, mesh_quat_w, ray_starts, ray_dirs, mesh_indices=None, description=""):
        print(f"Benchmarking {description}...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        mem_before = torch.cuda.memory_allocated()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        method = getattr(raycaster, method_name)
        kwargs = {
            "mesh_pos_w": mesh_pos_w,
            "mesh_quat_w": mesh_quat_w,
            "ray_starts_w": ray_starts.clone(),
            "ray_dirs_w": ray_dirs.clone(),
            "min_dist": 0.05,
            "max_dist": 100.0,
        }
        if mesh_indices is not None:
            kwargs["mesh_indices"] = mesh_indices
        
        start_event.record()
        for _ in range(num_iterations):
            hit_positions, hit_distances = method(**kwargs)
            wp.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / num_iterations
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = mem_after - mem_before
        
        return elapsed_time, mem_before, mem_after, mem_peak, mem_used, hit_positions, hit_distances
    
    # Benchmark all 4 combinations
    time_1_fused, mem_before_1_fused, mem_after_1_fused, mem_peak_1_fused, mem_used_1_fused, pos_1_fused, dist_1_fused = benchmark_raycast(
        raycaster_independent, "raycast_fused",
        translations_meshes,
        quaternions,
        ray_starts, ray_dirs,
        mesh_indices=mesh_indices,
        description="Option 1 (independent meshes) - FUSED"
    )
    
    # time_1_nonfused, mem_before_1_nonfused, mem_after_1_nonfused, mem_peak_1_nonfused, mem_used_1_nonfused, pos_1_nonfused, dist_1_nonfused = benchmark_raycast(
    #     raycaster_independent, "raycast",
    #     translations,
    #     quaternions,
    #     ray_starts, ray_dirs,
    #     mesh_indices=mesh_indices,
    #     description="Option 1 (independent meshes) - NON-FUSED"
    # )
    
    time_2_fused, mem_before_2_fused, mem_after_2_fused, mem_peak_2_fused, mem_used_2_fused, pos_2_fused, dist_2_fused = benchmark_raycast(
        raycaster_combined, "raycast_fused",
        torch.tensor([0., 0., 0.], device="cuda").expand(N * M + 1, 3),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N * M + 1, 4),
        ray_starts, ray_dirs,
        description="Option 2 (combined mesh) - FUSED"
    )
    
    breakpoint()
    # pos_2_fused = pos_1_fused
    # pos_1_fused = pos_2_fused
    
    # time_2_nonfused, mem_before_2_nonfused, mem_after_2_nonfused, mem_peak_2_nonfused, mem_used_2_nonfused, pos_2_nonfused, dist_2_nonfused = benchmark_raycast(
    #     raycaster_combined, "raycast",
    #     torch.tensor([0., 0., 0.], device="cuda").expand(N * M, 1, 3),
    #     torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N * M, 1, 4),
    #     ray_starts, ray_dirs,
    #     description="Option 2 (combined mesh) - NON-FUSED"
    # )
    
    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    print("\nTIMING (ms per iteration):")
    print(f"  Option 1 (independent meshes) - FUSED:     {time_1_fused:.4f} ms")
    # print(f"  Option 1 (independent meshes) - NON-FUSED: {time_1_nonfused:.4f} ms")
    print(f"  Option 2 (combined mesh) - FUSED:          {time_2_fused:.4f} ms")
    # print(f"  Option 2 (combined mesh) - NON-FUSED:       {time_2_nonfused:.4f} ms")
    
    # print("\nFUSED vs NON-FUSED Comparison:")
    # print(f"  Option 1: Fused is {time_1_nonfused / time_1_fused:.2f}x faster" if time_1_fused < time_1_nonfused else f"  Option 1: Non-fused is {time_1_fused / time_1_nonfused:.2f}x faster")
    # print(f"  Option 2: Fused is {time_2_nonfused / time_2_fused:.2f}x faster" if time_2_fused < time_2_nonfused else f"  Option 2: Non-fused is {time_2_fused / time_2_nonfused:.2f}x faster")
    
    print("\nOPTION 1 vs OPTION 2 Comparison:")
    print(f"  Fused:   Option 1 is {time_2_fused / time_1_fused:.2f}x faster" if time_1_fused < time_2_fused else f"  Fused:   Option 2 is {time_1_fused / time_2_fused:.2f}x faster")
    # print(f"  Non-fused: Option 1 is {time_2_nonfused / time_1_nonfused:.2f}x faster" if time_1_nonfused < time_2_nonfused else f"  Non-fused: Option 2 is {time_1_nonfused / time_2_nonfused:.2f}x faster")
    
    print("\nMEMORY USAGE:")
    print(f"  Option 1 (independent meshes) - FUSED:")
    print(f"    Peak memory: {format_memory(mem_peak_1_fused)}")
    print(f"    Memory used: {format_memory(mem_used_1_fused)}")
    # print(f"  Option 1 (independent meshes) - NON-FUSED:")
    # print(f"    Peak memory: {format_memory(mem_peak_1_nonfused)}")
    # print(f"    Memory used: {format_memory(mem_used_1_nonfused)}")
    print(f"  Option 2 (combined mesh) - FUSED:")
    print(f"    Peak memory: {format_memory(mem_peak_2_fused)}")
    print(f"    Memory used: {format_memory(mem_used_2_fused)}")
    # print(f"  Option 2 (combined mesh) - NON-FUSED:")
    # print(f"    Peak memory: {format_memory(mem_peak_2_nonfused)}")
    # print(f"    Memory used: {format_memory(mem_used_2_nonfused)}")
    
    # print("\nMEMORY COMPARISON:")
    # if mem_peak_1_fused < mem_peak_1_nonfused:
    #     print(f"  Option 1: Fused uses {mem_peak_1_nonfused / mem_peak_1_fused:.2f}x less peak memory")
    # else:
    #     print(f"  Option 1: Non-fused uses {mem_peak_1_fused / mem_peak_1_nonfused:.2f}x less peak memory")
    # if mem_peak_2_fused < mem_peak_2_nonfused:
    #     print(f"  Option 2: Fused uses {mem_peak_2_nonfused / mem_peak_2_fused:.2f}x less peak memory")
    # else:
    #     print(f"  Option 2: Non-fused uses {mem_peak_2_fused / mem_peak_2_nonfused:.2f}x less peak memory")
    
    # Compare Option 1 vs Option 2 (using fused versions)
    if mem_peak_1_fused < mem_peak_2_fused:
        print(f"  Option 1 vs Option 2 (fused): Option 1 uses {mem_peak_2_fused / mem_peak_1_fused:.2f}x less peak memory")
    else:
        print(f"  Option 1 vs Option 2 (fused): Option 2 uses {mem_peak_1_fused / mem_peak_2_fused:.2f}x less peak memory")
    
    print("="*70)
    
    # Verify results are similar
    print("\nVerifying results match...")
    results = [
        ("Option 1 Fused", pos_1_fused, dist_1_fused),
        # ("Option 1 Non-fused", pos_1_nonfused, dist_1_nonfused),
        ("Option 2 Fused", pos_2_fused, dist_2_fused),
        # ("Option 2 Non-fused", pos_2_nonfused, dist_2_nonfused),
    ]
    
    # Compare all against Option 1 Fused as reference
    ref_name, ref_pos, ref_dist = results[0]
    
    # if N * M <= 64:
    if True:
        scene = trimesh.Scene()
        # Make shapes_combined slightly transparent
        shapes_combined.visual.face_colors = [128, 128, 128, 200]  # Gray color with slight transparency
        scene.add_geometry(shapes_combined)
        pcd = trimesh.PointCloud(pos_1_fused.reshape(-1, 3).cpu().numpy(), colors=[0., 0., 1.])
        scene.add_geometry(pcd, node_name="Option 1 Fused")
        pcd = trimesh.PointCloud(pos_2_fused.reshape(-1, 3).cpu().numpy(), colors=[0., 1., 0.])
        scene.add_geometry(pcd, node_name="Option 2 Fused")
        segments = torch.stack([ray_starts.reshape(-1, 3), pos_2_fused.reshape(-1, 3)], dim=1).cpu().numpy()
        lines = trimesh.load_path(segments)
        scene.add_geometry(lines, node_name="Option 2 Fused Segments")
        scene.show()

    for name, pos, dist in results[1:]:
        pos_match = torch.allclose(ref_pos, pos, atol=1e-5)
        dist_match = torch.allclose(ref_dist, dist, atol=1e-5)
        
        if pos_match and dist_match:
            print(f"✓ {name} matches {ref_name}")
        else:
            print(f"✗ {name} differs from {ref_name}")
            if not pos_match:
                print(f"    Positions - Max diff: {(ref_pos - pos).abs().max().item():.6f}, Mean diff: {(ref_pos - pos).abs().mean().item():.6f}")
            if not dist_match:
                print(f"    Distances - Max diff: {(ref_dist - dist).abs().max().item():.6f}, Mean diff: {(ref_dist - dist).abs().mean().item():.6f}")


if __name__ == "__main__":
    main()

