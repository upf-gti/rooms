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

@group(0) @binding(0) var<storage, read> num_brinks_by_workgroup : u32;
@group(0) @binding(2) var<storage, read_write> bricks_to_write_to_tex_count : atomic<i32>;
@group(0) @binding(4) var<storage, read_write> bricks_to_write_to_tex_buffer : array<u32>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

#include sdf_functions.wgsl

var<workgroup> wg_used_pixels : atomic<u32>;
var<workgroup> wg_current_brick_to_process : i32;
var<workgroup> wg_octree_leaf_id : u32;
var<workgroup> wg_brick_pointer : u32;
var<workgroup> wg_brick_index : u32;
var<workgroup> wg_brick_center : vec3f;
var<workgroup> wg_atlas_tile_coord : vec3u;

/**
    Este shader se ejecuta despues de la ultima pasada del evaluador, y su funcion es evaluar y escribir en el
    Atlas 3D la SDF, para que pueda ser renderizada mas eficientemente.
    Este shader utiliza llamadas de work group de tamano de 8x8x8, ya que cada hilo escribe un texel
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
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(local_invocation_index) thread_id: u32)
{
    // PROTO:
    let local_id_vec : vec3f = vec3f(local_id);
    // Offset to cover negative and positive
    // for example, for threads going from 0 to 7 -> -3.5 to 3.5
    const offset : f32 = (ATLAS_BRICK_SIZE - 1.0) * 0.5;
    let pixel_offset : vec3f = (local_id_vec - offset) * BRICK_VOXEL_WORLD_SIZE;

    for(var i : u32 = 0u; i < num_brinks_by_workgroup; i++) {
        var result_surface : Surface;

        if (thread_id == 0) {
            let current_brick_to_do : i32 = atomicSub(&bricks_to_write_to_tex_count, 1) - 1;

            wg_current_brick_to_process = current_brick_to_do;

            // let culling_count : u32 = octree.data[octree_leaf_id].stroke_count;
            // let curr_culling_layer_index = octree.data[octree_leaf_id].culling_id;
            wg_octree_leaf_id = bricks_to_write_to_tex_buffer[wg_current_brick_to_process];
            wg_brick_pointer = octree.data[wg_octree_leaf_id].tile_pointer;

            // Get the brick index, without the MSb that signals if it has an already initialized brick
            let brick_index : u32 = wg_brick_pointer & OCTREE_TILE_INDEX_MASK;

            wg_brick_index = brick_index;

            let proxy_data : ptr<storage, ProxyInstanceData, read_write> = &brick_buffers.brick_instance_data[brick_index];
            wg_brick_center = proxy_data.position;
            let atlas_tile_index : u32 = proxy_data.atlas_tile_index;
            // Get the 3D atlas coords of the brick, with a stride of ATLAS_BRICK_SIZE
            wg_atlas_tile_coord = u32(ATLAS_BRICK_SIZE) * vec3u(atlas_tile_index % NUM_BRICKS_IN_ATLAS_AXIS,
                                                        (atlas_tile_index / NUM_BRICKS_IN_ATLAS_AXIS) % NUM_BRICKS_IN_ATLAS_AXIS,
                                                        atlas_tile_index / (NUM_BRICKS_IN_ATLAS_AXIS * NUM_BRICKS_IN_ATLAS_AXIS));

            atomicStore(&wg_used_pixels, 0u);
        }

        workgroupBarrier();

        if (wg_current_brick_to_process > 0) {
            // Init baking the SDF to the texture
            let pos = wg_brick_center + pixel_offset;

            let is_interior_brick : bool = (INTERIOR_BRICK_FLAG & wg_brick_pointer) == INTERIOR_BRICK_FLAG;

            if (is_interior_brick) {
                result_surface.distance = -100.0;
            } else {
                result_surface.distance = 10000.0;
            }

            // SDF compute per pixel
            let stroke_count : u32 = stroke_history.count;
            for(var i : u32 = 0u; i < stroke_count; i++) {
                result_surface = evaluate_stroke(pos, &(stroke_history.strokes[i]), &edit_list, result_surface, stroke_history.strokes[i].edit_list_index, stroke_history.strokes[i].edit_count);
            }

            // validate if there is something in hte brick
            if (result_surface.distance < MIN_HIT_DIST) {
                atomicAdd(&wg_used_pixels, 1);
            }

            let texture_coordinates : vec3u = wg_atlas_tile_coord + local_id;

            // Note, maybe this can be moved before the barrier, to avoid storing on a empty block
            textureStore(write_sdf, texture_coordinates, vec4f(result_surface.distance));
            textureStore(write_material_sdf, texture_coordinates, vec4<u32>((pack_material(result_surface.material))));
        }

        workgroupBarrier();

        // Check if the current brick is empty
        if (thread_id == 0u && wg_current_brick_to_process > 0) {
            let filled_pixel_count : u32 = atomicLoad(&wg_used_pixels);
            if (filled_pixel_count == 0u || filled_pixel_count == 1000u) {
                
                brick_buffers.brick_instance_data[wg_brick_index].in_use = 0;
                // Add the brick to the indirect
                let brick_to_delete_idx = atomicAdd(&brick_buffers.brick_removal_counter, 1u);
                brick_buffers.brick_removal_buffer[brick_to_delete_idx] = wg_brick_index;

                if (filled_pixel_count == 1000u) {
                    octree.data[wg_octree_leaf_id].octant_center_distance = vec2f(-10000.0, -10000.0);
                    octree.data[wg_octree_leaf_id].tile_pointer = INTERIOR_BRICK_FLAG;
                } else {
                    octree.data[wg_octree_leaf_id].octant_center_distance = vec2f(10000.0, 10000.0);
                    octree.data[wg_octree_leaf_id].tile_pointer = 0u;
                }

                let t = stroke_culling[0];
            } 
        }

        // NOTE: version of the prev block that only stores in teh atlas the non-emtpy bricks
        // if (wg_current_brick_to_process > 0u) {
        //     let filled_pixel_count : u32 = atomicLoad(&wg_used_pixels);
        //     // If its empty (0 pixels with surface) or full (512 piixels with surface),
        //     // We do not store the values to the atlas, and the trhead_0 adds it to the 
        //     // empty brick pile
        //     if (filled_pixel_count == 0u || filled_pixel_count == 512u) {
        //         if (thread_id == 0u) {
        //             // Remove the brick
        //             brick_buffers.brick_instance_data[brick_index].in_use = 0;
        //             // Add the brick to the indirect
        //             let brick_to_delete_idx = atomicAdd(&brick_buffers.brick_removal_counter, 1u);
        //             brick_buffers.brick_removal_buffer[brick_to_delete_idx] = brick_index;

        //             if (filled_pixel_count == 512u) {
        //                 octree.data[wg_octree_leaf_id].wg_brick_center_distance = vec2f(-10000.0, -10000.0);
        //                 octree.data[wg_octree_leaf_id].tile_pointer = INTERIOR_BRICK_FLAG;
        //             } else {
        //                 octree.data[wg_octree_leaf_id].wg_brick_center_distance = vec2f(10000.0, 10000.0);
        //                 octree.data[wg_octree_leaf_id].tile_pointer = 0u;
        //             }
        //         }
        //     } else {
        //         textureStore(write_sdf, texture_coordinates, vec4f(result_surface.distance));
        //         textureStore(write_material_sdf, texture_coordinates, vec4<u32>((pack_material(result_surface.material))));
        //     }
        // }
        
    }

    // // Evaluating the edit context
    // //let p = stroke_culling[0];
    // for (var j : u32 = 0; j < culling_count; j++) {
    //     let index : u32 = culling_get_stroke_index(stroke_culling[j + curr_culling_layer_index]);
    //     curr_surface = evaluate_stroke(pos, &(stroke_history.strokes[index]), &edit_list, curr_surface, stroke_history.strokes[index].edit_list_index, stroke_history.strokes[index].edit_count);
    // }

    // // Non-culled part
    // let culled_part : u32 = (min(stroke_history.count, MAX_STROKE_INFLUENCE_COUNT));
    // let non_culled_count : u32 = ( (stroke_history.count) - culled_part);
    // for(var i : u32 = 0u; i < non_culled_count; i++) {
    //     let index : u32 = i + MAX_STROKE_INFLUENCE_COUNT;
    //     curr_surface = evaluate_stroke(pos, &(stroke_history.strokes[index]), &edit_list, curr_surface, stroke_history.strokes[index].edit_list_index, stroke_history.strokes[index].edit_count);
    // }
}
