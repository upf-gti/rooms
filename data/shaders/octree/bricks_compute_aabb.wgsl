#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> aabb_group_storage : array<sPaddedAABB>;
@group(0) @binding(1) var<storage, read_write> brick_index_buffer : array<u32>;

@group(1) @binding(0) var<storage, read_write> gpu_return_results: GPUReturnResults;

/**
    Single workgroup reduce using local memory
*/

var<workgroup> shared_AABB_buffer : array<sPaddedAABB, 64u>;

fn merge_padded_aabb(aabb1 : sPaddedAABB, aabb2 : sPaddedAABB) -> sPaddedAABB {
    var result : sPaddedAABB;
    result.min = min(aabb1.min, aabb2.min);
    result.max = min(aabb1.max, aabb2.max);

    return result;
}

@compute @workgroup_size(4u, 4u, 4u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    var local_AABB : sPaddedAABB;

    let starting_idx : u32 = dot(id, vec3u(1u, 4u, 16u)) * 512u;

    // Reduce 512 -> 64
    for (var i : u32 = 0u; i < 8u; i++) {
        local_AABB = merge_padded_aabb(local_AABB, aabb_group_storage[starting_idx + local_id * 8u + i]);
    }

    // Store the shared memory
    shared_AABB_buffer[local_id] = local_AABB;

    workgroupBarrier();

    local_AABB = {};
    // Reduce 64 -> 16
    if (local_id < 16) {
        for (var i : u32 = 0u; i < 4u; i++) {
            local_AABB = merge_padded_aabb(local_AABB, shared_AABB_buffer[local_id * 4u + i]);
        }
    }

    workgroupBarrier();
    // Store to shared memory
    shared_AABB_buffer[local_id] = local_AABB;

    workgroupBarrier();

    local_AABB = {};

    // Reduce 64 -> 4
    if (local_id < 4) {
        for (var i : u32 = 0u; i < 4u; i++) {
            local_AABB = merge_padded_aabb(local_AABB, shared_AABB_buffer[local_id * 4u + i]);
        }
    }

    workgroupBarrier();
    // Store to shared memory
    shared_AABB_buffer[local_id] = local_AABB;

    workgroupBarrier();

    local_AABB = {};

    // Reduce 4 -> 1
    if (local_id == 0u) {
        for (var i : u32 = 0u; i < 4u; i++) {
            local_AABB = merge_padded_aabb(local_AABB, shared_AABB_buffer[local_id + i]);
        }

        // Store the AABB in the read-back struct
        atomicMin(&gpu_return_results.sculpt_aabb_min, local_AABB.min);
        atomicMax(&gpu_return_results.sculpt_aabb_max, local_AABB.max);
    }
}
