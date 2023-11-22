#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> brick_copy_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> proxy_box_indirect : OctreeProxyIndirect;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data : OctreeProxyInstances;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let current_instance_index : u32 = id.x;

    let current_instance : ProxyInstanceData = octree_proxy_data.instance_data[current_instance_index];

    if (current_instance.in_use == 1) {
        let prev_value : u32 = atomicAdd(&proxy_box_indirect.instance_count, 1u);
        brick_copy_buffer[prev_value] = current_instance_index;
    }
}
