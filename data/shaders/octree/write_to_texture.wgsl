#include math.wgsl
#include sdf_utils.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include ../noise.wgsl

@group(0) @binding(3) var write_sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory; 
@group(0) @binding(7) var<storage, read> edit_list : array<Edit>;
@group(0) @binding(8) var write_material_sdf: texture_storage_3d<r32uint, write>;
@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

#include sdf_functions.wgsl

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

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;
    let octree_leaf_id : u32 = octant_usage_read[id];

    let culling_count : u32 = octree.data[octree_leaf_id].stroke_count;
    let curr_culling_layer_index = octree.data[octree_leaf_id].culling_id;

    let brick_pointer : u32 = octree.data[octree_leaf_id].tile_pointer;

    // Get the brick index, without the MSb that signals if it has an already initialized brick
    // 3	 v_and_b32_e32	 v0, 0x3ff, v0	 814	 0.01	 85 clk	 
    let brick_index : u32 = brick_pointer & OCTREE_TILE_INDEX_MASK;

    let proxy_data : ProxyInstanceData = brick_buffers.brick_instance_data[brick_index];
    let local_id_vec : vec3f = vec3f(local_id);

    // let voxel_world_coords : vec3f = proxy_data.position + (ATLAS_BRICK_SIZE / local_id_vec - (ATLAS_BRICK_SIZE * 0.5)) * BRICK_VOXEL_WORLD_SIZE;

    // Get the 3D atlas coords of the brick, with a stride of ATLAS_BRICK_SIZE
    let atlas_tile_coordinate : vec3u = u32(ATLAS_BRICK_SIZE) * vec3u(proxy_data.atlas_tile_index % NUM_BRICKS_IN_ATLAS_AXIS,
                                                  (proxy_data.atlas_tile_index / NUM_BRICKS_IN_ATLAS_AXIS) % NUM_BRICKS_IN_ATLAS_AXIS,
                                                   proxy_data.atlas_tile_index / (NUM_BRICKS_IN_ATLAS_AXIS * NUM_BRICKS_IN_ATLAS_AXIS));
    
    let octant_center : vec3f = proxy_data.position;

    var sSurface : Surface;

    let texture_coordinates : vec3u = atlas_tile_coordinate + local_id;

    let is_interior_brick : bool = (INTERIOR_BRICK_FLAG & brick_pointer) == INTERIOR_BRICK_FLAG;

    if (is_interior_brick) {
        sSurface.distance = -100.0;
    } else {
        sSurface.distance = 10000.0;
    }

    // Offset to cover negative and positive
    // for example, for threads going from 0 to 7 -> -3.5 to 3.5
    const offset : f32 = (ATLAS_BRICK_SIZE - 1.0) * 0.5;
    let pixel_offset : vec3f = (local_id_vec - offset) * BRICK_VOXEL_WORLD_SIZE;

    var result_surface : Surface;
    result_surface.distance = 0.0;

    var curr_surface : Surface = sSurface;
    let pos = octant_center + pixel_offset;

    // Evaluating the edit context
    //let p = stroke_culling[0];
    for (var j : u32 = 0; j < culling_count; j++) {
        let index : u32 = culling_get_stroke_index(stroke_culling[j + curr_culling_layer_index]);
        curr_surface = evaluate_stroke(pos, &(stroke_history.strokes[index]), &edit_list, curr_surface, stroke_history.strokes[index].edit_list_index, stroke_history.strokes[index].edit_count);
    }

    // Non-culled part
    let culled_part : u32 = (min(stroke_history.count, MAX_STROKE_INFLUENCE_COUNT));
    let non_culled_count : u32 = ( (stroke_history.count) - culled_part);
    for(var i : u32 = 0u; i < non_culled_count; i++) {
        let index : u32 = i + MAX_STROKE_INFLUENCE_COUNT;
        curr_surface = evaluate_stroke(pos, &(stroke_history.strokes[index]), &edit_list, curr_surface, stroke_history.strokes[index].edit_list_index, stroke_history.strokes[index].edit_count);
    }

    result_surface = curr_surface;

    //wtt: 1563
    if (result_surface.distance < MIN_HIT_DIST) {
        atomicAdd(&used_pixels, 1);
    }

    //result_surface.material.albedo = vec3f(f32(culling_count)/ 15.0);

    // Duplicate the texture Store, becuase then we have a branch depeding on an uniform!
    textureStore(write_sdf, texture_coordinates, vec4f(result_surface.distance));
    textureStore(write_material_sdf, texture_coordinates, vec4<u32>((pack_material(result_surface.material))));
    
    // Hack, for buffer usage
    octant_usage_write[0] = 0;

    workgroupBarrier();

    if (local_id.x == 0 && local_id.y == 0 && local_id.z == 0) {

        let filled_pixel_count : u32 = atomicLoad(&used_pixels);
        if (filled_pixel_count == 0u || filled_pixel_count == 1000u) {
            
            brick_buffers.brick_instance_data[brick_index].in_use = 0;
            // Add the brick to the indirect
            let brick_to_delete_idx = atomicAdd(&brick_buffers.brick_removal_counter, 1u);
            brick_buffers.brick_removal_buffer[brick_to_delete_idx] = brick_index;

            if (filled_pixel_count == 1000u) {
                octree.data[octree_leaf_id].octant_center_distance = vec2f(-10000.0, -10000.0);
                octree.data[octree_leaf_id].tile_pointer = INTERIOR_BRICK_FLAG;
            } else {
                octree.data[octree_leaf_id].octant_center_distance = vec2f(10000.0, 10000.0);
                octree.data[octree_leaf_id].tile_pointer = 0u;
            }

        } 
        // else {
        //     // Add "filled" flag and remove "interior" flag
        //     octree.data[octree_leaf_id].tile_pointer = brick_index | FILLED_BRICK_FLAG;
        // }

    }
}
