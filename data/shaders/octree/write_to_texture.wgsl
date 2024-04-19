#include ../math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include ../noise.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var write_sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory; 
@group(0) @binding(7) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;
@group(0) @binding(8) var write_material_sdf: texture_storage_3d<r32uint, read_write>;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

var<workgroup> used_pixels : atomic<u32>;

/**
    Este shader se ejecuta despues de la ultima pasada del evaluador, y su funcion es evaluar y escribir en el
    Atlas 3D la SDF, para que pueda ser renderizada mas eficientemente.
    Este shader utiliza llamadas de work group de tamano de 10x10x10, ya que cada hilo escribe un texel
    en el Atlas.
    En cada hilo, evaluamos primero el "contexto" del stroke actual (que es todos los edits a su alrededor que 
    pueden influir) para luego evaular el stroke actual. Esto es importante para que la evaluacion Smooth 
    influencie correctamente a las SDFs de alrededor y las del entorno influyan a la nueva SDF.
    Este shader escribe a las 2 texturas del Atlas, a la que almacena distancias y la que almacena materiales.
    
    Despues de evaluar todos los threads, el shader cuenta cuantos texeles se han escrito dentro o fuera
    de la superficie, y si el resultado es o todos (1000 texeles) o ninguno (0 texeles), el brick no tiene
    superficeie y se marca para su borrado. Esto puede pasar por problemas de precision de los intervalos
    del evaluador.
*/

// fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
//     return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
// }

// const delta_pos_world = array<vec3f, 9>(
//     vec3f(PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(-PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(-PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(-PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(-PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
//     vec3f(0.0, 0.0, 0.0),
// );

@compute @workgroup_size(10,10,10)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;
    let octree_leaf_id : u32 = octant_usage_read[id];

    let brick_pointer : u32 = octree.data[octree_leaf_id].tile_pointer;

    // Get the brick index, without the MSb that signals if it has an already initialized brick
    // 3	 v_and_b32_e32	 v0, 0x3ff, v0	 814	 0.01	 85 clk	 
    let brick_index : u32 = brick_pointer & OCTREE_TILE_INDEX_MASK;

    let proxy_data : ProxyInstanceData = octree_proxy_data.instance_data[brick_index];
    let local_id_vec : vec3f = vec3f(local_id);

    let voxel_world_coords : vec3f = proxy_data.position + (10.0 / local_id_vec - 5.0) * PIXEL_WORLD_SIZE;

    // Get the 3D atlas coords of the brick, with a stride of 10 (the size of the brick)
    let atlas_tile_coordinate : vec3u = 10 * vec3u(proxy_data.atlas_tile_index % BRICK_COUNT,
                                                  (proxy_data.atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   proxy_data.atlas_tile_index / (BRICK_COUNT * BRICK_COUNT));
    
    let octant_center : vec3f = proxy_data.position;

    var sSurface : Surface;

    let texture_coordinates : vec3u = atlas_tile_coordinate + local_id;

    var material : Material;
    material.albedo = stroke.material.color.xyz;
    material.roughness = stroke.material.roughness;
    material.metalness = stroke.material.metallic;

    // wtt: BBF0_0 17
    if ((INTERIOR_BRICK_FLAG & brick_pointer) == INTERIOR_BRICK_FLAG) {
        sSurface.distance = -100.0;
    } else {
        sSurface.distance = 10000.0;
    }
    // Offset for a 10 pixel wide brick
    let pixel_offset : vec3f = (local_id_vec - 4.5) * PIXEL_WORLD_SIZE;

    var result_surface : Surface;
    result_surface.distance = 0.0;


    var curr_surface : Surface = sSurface;
    let pos = octant_center + pixel_offset;

    // Evaluating the edit context
    for (var j : u32 = 0; j < stroke_history.count; j++) {
        curr_surface = evaluate_edit_2(pos, &(stroke_history.strokes[j]), curr_surface);
    }

    // Evaluate current stroke
    curr_surface = evaluate_edit_2(pos, &(stroke), curr_surface);

    result_surface = curr_surface;

    //wtt: 1563
    if (result_surface.distance < MIN_HIT_DIST) {
        atomicAdd(&used_pixels, 1);
    }

    // Duplicate the texture Store, becuase then we have a branch depeding on an uniform!
    textureStore(write_sdf, texture_coordinates, vec4f(result_surface.distance));
    textureStore(write_material_sdf, texture_coordinates, vec4<u32>((pack_material(result_surface.material))));
    
    // Hack, for buffer usage
    octant_usage_write[0] = 0;
    let edit_count : u32 = stroke_history.count;

    workgroupBarrier();

    if (local_id.x == 0 && local_id.y == 0 && local_id.z == 0) {

        let filled_pixel_count : u32 = atomicLoad(&used_pixels);
        if (filled_pixel_count == 0u || filled_pixel_count == 1000u) {
            
            octree_proxy_data.instance_data[brick_index].in_use = 0;
            // Add the brick to the indirect
            let brick_to_delete_idx = atomicAdd(&indirect_brick_removal.brick_removal_counter, 1u);
            indirect_brick_removal.brick_removal_buffer[brick_to_delete_idx] = brick_index;

            octree.data[octree_leaf_id].octant_center_distance = vec2f(10000.0, 10000.0);
            octree.data[octree_leaf_id].tile_pointer = 0u;
        } else {
            // Add "filled" flag and remove "interior" flag
            octree.data[octree_leaf_id].tile_pointer = brick_index | FILLED_BRICK_FLAG;
        }

    }

    // octree.data[octree_leaf_id].tile_pointer = brick_index | FILLED_BRICK_FLAG;
    // let brick_to_delete_idx = indirect_brick_removal.indirect_padding;
}
