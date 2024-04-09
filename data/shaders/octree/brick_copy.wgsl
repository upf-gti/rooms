#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> brick_copy_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> proxy_box_indirect : OctreeProxyIndirect;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data : OctreeProxyInstances;

/**
    Este shader itera por todo el buffer de render instances de bricks, y en funcion
    de si esta lleno o vacio, incrementa el numero de instancias del indirect buffer
    para el instancing de los bricks.
    Tambien se rellena un buffer que relaciona cada instancia con un indice del instancing.
*/

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = (id.x) * 512u + local_id;

    let current_instance : ProxyInstanceData = octree_proxy_data.instance_data[current_instance_index];

    if ((current_instance.in_use & BRICK_IN_USE_FLAG) == BRICK_IN_USE_FLAG) {
        let prev_value : u32 = atomicAdd(&proxy_box_indirect.instance_count, 1u);
        brick_copy_buffer[prev_value] = current_instance_index;
    }
}
