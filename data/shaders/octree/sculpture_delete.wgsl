#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> octree : Octree_NonAtomic;
@group(1) @binding(0) var<storage, read_write> brick_buffers: BrickBuffers;

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = (id.x) * 512u + local_id;

    let octree_index : u32 = current_instance_index + u32((pow(8.0, f32(OCTREE_DEPTH)) - 1) / 7);

    if ((octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
        let brick_index : u32 = octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK;
        let index : u32 = atomicAdd(&brick_buffers.atlas_empty_bricks_counter, 1u);
        brick_buffers.atlas_empty_bricks_buffer[index] = brick_buffers.brick_instance_data[brick_index].atlas_tile_index;
        
        brick_buffers.brick_instance_data[brick_index].in_use = 0u;
        brick_buffers.brick_instance_data[brick_index].atlas_tile_index = 0u;
    }
}