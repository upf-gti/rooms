#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(4) var<storage, read_write> counters : OctreeCounters;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let num_dispatches : u32 = atomicLoad(&counters.atomic_counter);

    indirect_buffer.x = num_dispatches;

    atomicAdd(&counters.current_level, 1);

    atomicStore(&counters.atomic_counter, 0);
}
