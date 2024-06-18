#include octree_includes.wgsl

struct SculptInstanceData {
    model_instances : array<mat4x4f>,
};

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
@group(0) @binding(0) var<storage, read_write> brick_copy_buffer : array<u32>;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;

@group(1) @binding(0) var<storage, read_write> sculpt_instances_buffer: array<u32>;

/**
    Este shader itera por todo el buffer de render instances de bricks, y en funcion
    de si esta lleno o vacio, incrementa el numero de instancias del indirect buffer
    para el instancing de los bricks.
    Tambien se rellena un buffer que relaciona cada instancia con un indice del instancing.
*/

fn get_buffer_index(instance_index : u32, model_index : u32) -> u32 {
    return (instance_index << 20) | model_index;
}

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = (id.x) * 512u + local_id;

    let current_instance_in_use_flag : u32 = brick_buffers.brick_instance_data[current_instance_index].in_use;
    let current_octree_index : u32 = brick_buffers.brick_instance_data[current_instance_index].octree_id;

    if ((current_instance_in_use_flag & BRICK_IN_USE_FLAG) == BRICK_IN_USE_FLAG
     && (current_instance_in_use_flag & BRICK_HIDE_FLAG) == 0)
    {
        let raw_instance_data : u32 = sculpt_instances_buffer[current_octree_index];
        let sculpt_instance_count : u32 = (raw_instance_data) >> 20u;
        let model_buffer_starting_index : u32 = (raw_instance_data & 0xFFFFF);

        for(var i : u32 = 0u; i < sculpt_instance_count; i++) {
            let prev_value : u32 = atomicAdd(&brick_buffers.brick_instance_counter, 1u);
            brick_copy_buffer[prev_value] = get_buffer_index(current_instance_index, sculpt_instances_buffer[model_buffer_starting_index + i]);
        }
    }
}
