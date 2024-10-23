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

//@group(1) @binding(1) var<storage, read_write> aabb_group_storage : array<sPaddedAABB>;

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

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = OCTREE_LAST_LEVEL_STARTING_IDX + (id.x) * 512u + local_id;

    let current_instance_in_use_flag : u32 = octree.data[current_instance_index].tile_pointer;

    if ((current_instance_in_use_flag & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG)
        //&& (current_instance_in_use_flag & BRICK_HIDE_FLAG) == 0)
    {
        let prev_index : u32 = atomicAdd(&sculpt_indirect.brick_count, 1u);

        brick_index_buffer[prev_index] = current_instance_in_use_flag & OCTREE_TILE_INDEX_MASK;
    }
}
