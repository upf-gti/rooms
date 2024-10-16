#include ray_intersection_includes.wgsl
#include octree_includes.wgsl

@group(0) @binding(3) var<storage, read_write> ray_intersection_info: RayIntersectionInfo;
@group(1) @binding(0) var<storage, read_write> gpu_return_results: GPUReturnResults;


@compute @workgroup_size(1, 1, 1)
fn compute()
{
    // Store the values on the buffer to be read from GPU
    gpu_return_results.ray_has_intersected = ray_intersection_info.intersected;
    gpu_return_results.ray_tile_pointer = ray_intersection_info.tile_pointer;
    gpu_return_results.ray_sculpt_id = ray_intersection_info.sculpt_id;
    gpu_return_results.ray_t = ray_intersection_info.ray_t;

    // Clean the buffers for the next frame
    ray_intersection_info.intersected = 0u;
    ray_intersection_info.tile_pointer = 0u;
    ray_intersection_info.sculpt_id = 0u;
    ray_intersection_info.ray_t = -0xFFFFFFFFf; // -0x1.fffffep+127f this better?
}