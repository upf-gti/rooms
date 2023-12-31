#include octree_includes.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(6) var<storage, read_write> proxy_box_indirect : OctreeProxyIndirect;
@group(0) @binding(8) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;


@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    let index_linear : u32 = (id.x) * (8u * 8u * 8u) + local_id;

    if (index_linear == 0) {
        atomicStore(&indirect_brick_removal.brick_removal_counter, 0);
        atomicStore(&proxy_box_indirect.instance_count, 0);
    }

    workgroupBarrier();

    let octree_index : u32 = index_linear + u32((pow(8.0, f32(6.0)) - 1) / 7);

    if ((octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
        let instance_id : u32 = octree.data[octree_index].tile_pointer & 0x3FFFFFFFu;
        let brick_to_delete_idx = atomicAdd(&indirect_brick_removal.brick_removal_counter, 1u);
        indirect_brick_removal.brick_removal_buffer[brick_to_delete_idx] = instance_id;
    }

    octree.data[octree_index].tile_pointer = 0u;
    octree.data[octree_index].octant_center_distance = vec2f(10000.0, 10000.0); 
}
