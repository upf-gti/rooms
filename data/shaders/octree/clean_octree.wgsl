#include octree_includes.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers;

/**
    Cuando despues de hacer varios undo, y llegas al final, este shader se encarga de limpiar el octree
    y vaciar los bricks que queden, para dejarlo todo como al arrancar el editor.
*/

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    let thread_group_size : u32 = 8u*8u*8u;
    let index_linear : u32 = (id.x) * thread_group_size + local_id;

    if (index_linear == 0) {
        atomicStore(&indirect_buffers.brick_removal_counter, 0);
        atomicStore(&indirect_buffers.brick_instance_count, 0);
    }

    workgroupBarrier();

    if (index_linear < OCTREE_TOTAL_SIZE-1) {
        if ((octree.data[index_linear].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
            let instance_id : u32 = octree.data[index_linear].tile_pointer & OCTREE_TILE_INDEX_MASK;
            let brick_to_delete_idx = atomicAdd(&indirect_buffers.brick_removal_counter, 1u);
            brick_buffers.brick_removal_buffer[brick_to_delete_idx] = instance_id;
        }
        octree.data[index_linear].tile_pointer = 0u;
        octree.data[index_linear].octant_center_distance = vec2f(10000.0, 10000.0); 
    }
}
