#include octree_includes.wgsl
/*
Sculpt intances u32
    12 -> n number of instances (0-4096)
    20 -> index of the n indices of the model matrices (0-1048576)

    First part of the buffer, indexing using the scult instance u32
    Then its just an array of indices
*/
/*
Brick copy buffer u32
    20 -> Brick ID (0-1048576)
    12 -> model index (0-4096)
*/
@group(0) @binding(0) var<storage, read_write> octree : Octree_NonAtomic;
@group(0) @binding(1) var<storage, read_write> brick_index_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> sculpt_indirect : SculptIndirectCall;
@group(0) @binding(5) var<storage, read> brick_buffers: BrickBuffers_ReadOnly;

@group(1) @binding(0) var<storage, read_write> gpu_return_results: GPUReturnResults_Atomic;

/**
    INSTANCING proposal 
        1 brick idx buffer por sculpt

        1 Indirect Instanced drawcall per octree
            36, instance_count * brick_count, 0, 0
        
        1 Sculpt tiene N bricks y M instancias -> copy buffer size N y M * N indirect instancing
            En el vertex buffer isntance_id / M -> brick_idx y instance_id mod M = model_idx

    Futuro:
        Culling 

    Este shader itera por todo el buffer de render instances de bricks, y en funcion
    de si esta lleno o vacio, incrementa el numero de instancias del indirect buffer
    para el instancing de los bricks.
    Tambien se rellena un buffer que relaciona cada instancia con un indice del instancing.
*/

fn get_buffer_index(instance_index : u32, model_index : u32) -> u32 {
    return (instance_index << 12) | model_index;
}

fn merge_padded_aabb(aabb1 : sPaddedAABB, aabb2 : sPaddedAABB) -> sPaddedAABB {
    var result : sPaddedAABB;
    result.min = min(aabb1.min, aabb2.min);
    result.max = max(aabb1.max, aabb2.max);

    return result;
}

var<workgroup> shared_AABB_buffer : array<sPaddedAABB, 512u>;

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = OCTREE_LAST_LEVEL_STARTING_IDX + (id.x) * 512u + local_id;

    let current_instance_in_use_flag : u32 = octree.data[current_instance_index].tile_pointer;

    let brick_half_size : vec3f = vec3f(SCULPT_MAX_SIZE / pow(2.0, f32(OCTREE_DEPTH + 1)));

    var local_AABB : sPaddedAABB = sPaddedAABB(vec3f(2.0), 0u, vec3f(-2.0), 0u);

    if ((current_instance_in_use_flag & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG)
        //&& (current_instance_in_use_flag & BRICK_HIDE_FLAG) == 0)
    {
        let prev_index : u32 = atomicAdd(&sculpt_indirect.brick_count, 1u);

        let proxy_data_idx : u32 = current_instance_in_use_flag & OCTREE_TILE_INDEX_MASK;

        brick_index_buffer[prev_index] = proxy_data_idx;

        let instance_data : ptr<storage, ProxyInstanceData, read> = &brick_buffers.brick_instance_data[proxy_data_idx];

        let brick_position : vec3f = instance_data.position;

        local_AABB = sPaddedAABB(brick_position - brick_half_size, 1u, brick_position + brick_half_size, 0u);
    }    
    shared_AABB_buffer[local_id] = local_AABB;

    workgroupBarrier();

    // TODO: a workgroup map reduce scheme would be nice here
    // like in the unused bricks_compute_aabb file

    if (local_id == 0u) {
        for(var i : u32 = 1u; i < 512u; i++) {
            local_AABB = merge_padded_aabb(local_AABB, shared_AABB_buffer[i]);
        }

        // Use atomicMax and atomicMin to compute the resulting AABB from all workgroups

        // There is no f32 atomics, so we ensure that the sign is always the same, and compare
        // as signed ints
        atomicMax(&gpu_return_results.sculpt_aabb_max_x, bitcast<i32>(local_AABB.max.x + 5.0));
        atomicMax(&gpu_return_results.sculpt_aabb_max_y, bitcast<i32>(local_AABB.max.y + 5.0));
        atomicMax(&gpu_return_results.sculpt_aabb_max_z, bitcast<i32>(local_AABB.max.z + 5.0));

        atomicMin(&gpu_return_results.sculpt_aabb_min_x, bitcast<i32>(local_AABB.min.x + 5.0));
        atomicMin(&gpu_return_results.sculpt_aabb_min_y, bitcast<i32>(local_AABB.min.y + 5.0));
        atomicMin(&gpu_return_results.sculpt_aabb_min_z, bitcast<i32>(local_AABB.min.z + 5.0));
    }
}
