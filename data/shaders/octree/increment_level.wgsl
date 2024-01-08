#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(2) var<storage, read_write> octree : Octree;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let num_dispatches : u32 = atomicLoad(&octree.atomic_counter);

    indirect_buffer.x = num_dispatches;

    atomicAdd(&octree.current_level, 1);

    atomicStore(&octree.atomic_counter, 0);
}
