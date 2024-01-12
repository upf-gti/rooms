#include octree_includes.wgsl

@group(0) @binding(5) var<storage, read_write> octree_proxy_data : OctreeProxyInstances;

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    let current_instance_index : u32 = (id.x) * (8u * 8u * 8u) + local_id;

    if ((octree_proxy_data.instance_data[current_instance_index].in_use & BRICK_HAS_PREVIEW_FLAG) == BRICK_HAS_PREVIEW_FLAG) {
         octree_proxy_data.instance_data[current_instance_index].in_use &= ~BRICK_HAS_PREVIEW_FLAG;
    }
}
