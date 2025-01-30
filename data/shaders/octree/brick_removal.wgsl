#include octree_includes.wgsl

@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers_ReadOnly;

/**
    Este shader toma el listado de bricks marcados para borrar por el evaluator o el write_to_texture.
    Se tiene que hacer en otro paso, porque se pueden dar condiciones de carrera al hacerlo en el
    evaluador, porque se estananadiendo y borrando los bricks a la vez.

    Ese shader simplemente despuelve el indice del brick a la pool de indices vacios del Atlas 3D,
    y marca la render instance del brick como vacia.
*/

@compute @workgroup_size(1,1,1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    for(var i : u32 = 0u; i < brick_buffers.brick_removal_counter; i++) {
        let empty_index : u32 = brick_buffers.brick_removal_buffer[i];

        let index : u32 = brick_buffers.atlas_empty_bricks_counter;
        brick_buffers.atlas_empty_bricks_counter++;
        brick_buffers.atlas_empty_bricks_buffer[index] = brick_buffers.brick_instance_data[empty_index].atlas_tile_index;
        
        brick_buffers.brick_instance_data[empty_index].in_use = 0u;
        brick_buffers.brick_instance_data[empty_index].atlas_tile_index = 0u;
    }

    brick_buffers.brick_removal_counter = 0u;
    
}