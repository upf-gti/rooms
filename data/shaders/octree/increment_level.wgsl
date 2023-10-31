#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> proxy_box_indirect : array<u32, 4>;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;
@group(0) @binding(5) var<storage, read_write> atomic_counter : atomic<u32>;
@group(0) @binding(8) var<storage, read_write> atlas_tile_counter : atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let num_dispatches : u32 = atomicLoad(&atomic_counter);

    indirect_buffer.x = num_dispatches;

    let prev_level : u32 = atomicAdd(&current_level, 1);

    if (merge_data.max_octree_depth == prev_level) {
        atomicAdd(&atlas_tile_counter, num_dispatches);
        proxy_box_indirect[1] = atomicLoad(&atlas_tile_counter);
    } else {
        atomicStore(&atomic_counter, 0);
    }
}
