#include ../sdf_functions.wgsl

struct OctreeNode {
    tile_pointer : u32
}

struct Octree {
    data : array<OctreeNode>
};


struct MergeData {
    edits_aabb_start      : vec3<u32>,
    edits_to_process      : u32,
    sculpt_start_position : vec3f,
    max_octree_depth      : u32,
    sculpt_rotation       : vec4f
};

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> proxy_box_indirect : array<u32, 4>;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;
@group(0) @binding(5) var<storage, read_write> atomic_counter : atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let num_dispatches : u32 = atomicLoad(&atomic_counter);

    indirect_buffer.x = num_dispatches;

    atomicStore(&atomic_counter, 0);
    let prev_level : u32 = atomicAdd(&current_level, 1);

    if (merge_data.max_octree_depth == prev_level + 1) {
        proxy_box_indirect[1] = num_dispatches;
    }
}
