#include octree_includes.wgsl

@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;

/**
    Este shader se encarga de recorrer todas las render instances de los bricks de
    superficie y quitar la flag de que tiene preview, para que se renove en la siguiente
    pasada del evaluador para el preview.
*/

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    let current_instance_index : u32 = (id.x) * (8u * 8u * 8u) + local_id;

    brick_buffers.brick_instance_data[current_instance_index].in_use &= ~BRICK_HAS_PREVIEW_FLAG;
    //brick_buffers.brick_instance_data[current_instance_index].in_use &= ~BRICK_HIDE_FLAG;
}
