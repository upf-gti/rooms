#include octree_includes.wgsl

@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(8) var<storage, read> indirect_brick_removal : IndirectBrickRemoval_ReadOnly;

/**
    Este shader toma el listado de bricks marcados para borrar por el evaluator o el write_to_texture.
    Se tiene que hacer en otro paso, porque se pueden dar condiciones de carrera al hacerlo en el
    evaluador, porque se estananadiendo y borrando los bricks a la vez.

    Ese shader simplemente despuelve el indice del brick a la pool de indices vacios del Atlas 3D,
    y marca la render instance del brick como vacia.
*/

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    let empty_index : u32 = indirect_brick_removal.brick_removal_buffer[group_id.x];

    let index : u32 = atomicAdd(&octree_proxy_data.atlas_empty_bricks_counter, 1u);
    octree_proxy_data.atlas_empty_bricks_buffer[index] = octree_proxy_data.instance_data[empty_index].atlas_tile_index;
    
    octree_proxy_data.instance_data[empty_index].in_use = 0u;
    octree_proxy_data.instance_data[empty_index].atlas_tile_index = 0u;
}