#include ../sdf_functions.wgsl

struct OctreeNode {
    tile_pointer : u32
}

struct Octree {
    data : array<OctreeNode>
};

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;
@group(0) @binding(5) var<storage, read_write> atomic_counter : atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let num_dispatches : u32 = atomicLoad(&atomic_counter);

    indirect_buffer.x = num_dispatches;

    atomicStore(&atomic_counter, 0);
    atomicAdd(&current_level, 1);
}
