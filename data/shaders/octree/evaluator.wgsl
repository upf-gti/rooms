#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
@group(0) @binding(7) var<storage, read> edit_list : array<Edit>;
@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

@group(2) @binding(0) var<storage, read> preview_stroke : PreviewStroke;

#include sdf_interval_functions.wgsl

/*
    Este shader es el responsable de subdividir el espacio, para una evaluacion mas eficaz de SDFs.
    La estructra de subdivision es de forma de octree y la idea es ir suprimiendo zonas que no tienen superficie (dentro o fuera de la SDF).
    La subdivision se hace usando arithmetica de intervalos, y este shader se ejecutara una vez por cada "piso" de
    octree, hasta llegar a el tamano minimo de los "bricks" en la escena y mandara a escribir en el atlas 3D
    solo las areas que contengan superfice de la SDF.
    Al ser el encargado de determinar los bricks de la superficie, este shader se encarga de crear bricks y sus instancias,
    y tambien de marcarlos para ser borrados posteriormente.

    Tambien se usa para calcular el preview, subdividiendo de manera normal, pero se salta el paso del write_to_texture,
    ya que es algo que cambia frame a frame. Simplemente se rellenan las posiciones de las instancias de los bricks de preview
    y se marcan que bricks de nuestra superficie estan interactuando con el preview.

    La subdivision del octree se hace primero usando una evaluacion de los primitivivos del stroke a evualuar como si fuesen union, 
    con dimensiones incrementadas por el SMOOTH_MARGIN. Esto es para asegurar que al llegar al ultimo nivel de toda la superficie
    que ocuparia el stroke (y la superficie de la escultura sobre la que influencia), incluso si tenemos varios falsos positivos.

    En la ultima capa de la evaluacion hacemos una pseudo-subdivision y evaluacion de los strokes (usando la funcion y params tal y como
    vienen de CPU), usando intervalos mas pequeños, para evitar el wrapping effect que peuden tener las transformaciones de rotacion
    en intervalos. Esto hace que con intervalos mas pequeños el margen de imprecision que anade esta rotacion sea menor, y sea mas fiable
    para detectar falsos positivos en la subdivision.

    Una vez hemos pasado esa primera prueba para saber si hay superficie o no, pasamos a evaluar los intervalos. Cuando evaluas smooth SDF,
    con un smooth amrgin alto, el goop o la distorsion que se genera entre 2 cuerpos ocurre dentro de un margen de 2 veces el radio del smooth margin.
    Si bien, podemos pasar todos los bricks del radio a el write_to_texture, serian muchos falsos positivos y una perdida significativa de rendimiento.
    En el caso de smooth adition, no presenta muchos problemas, ya que simplemente mirando los bounds del intervalos podemos ver si esta en un brick o no,
    con cierta precision (el wrappinge ffect y formulaciones de intervalos, puede incrementar los margenes, lo cual casi siempre introducira falsos
    positivos).
    Pero en la substraccion smooth es mas dificil. Ya que la substraccion no tiene por si misma ninguna distancia, solo se puede medir sobre una
    superficie que ya existe y ver su cambio. Previamente, para solucionarlo lo tratabamos como una operacion de adicion sin smooth, y mirabamos 
    la influencia sobre los bricks en funcion de los margenes que esto nos da. Pero con las sdf smooth, esta informacion no es suficiente, ya 
    que por como afecta a las superficies de alrededor esto no es factible por si solo. La solucion es evaluar sobre los intervalos de la escultura Y
    usando una evaluacion smooth union sobre los mismos intervalos y comparandolos. Viendo la diferencia entre ellos podemos ver sobre que bricks la 
    smooth union ha actuado, y solo dentro del margen del stroke que se quiere procesar.
    TODO(Juan) NOTA: esto es posible que se pueda hacer SOLO comparando la diferencia de antes y despues aplicar el smooth substraction
    
    Octree Octant indexing
        - 3 bits per each layer, describes 8 octants.
        - With a u32 up to 10 layers -> This gives us indexing in an octree with 1024^3 leaves (10 layers).
        - The base layer (top of the tree) is common to all nodes, so the first layer is skipped in the respresentation.
        - We currently use 7 layers, due or bricks being 8x8x8.
        - The 3 less significant bits represent the octant index of the second layer.
        - The 3 more significant bits represetne the octant index of the leafs´s layer.
            | Leaves 3 bits | .... | 3rd layer 3 bits | 2nd layer 3 bits |
        - That way we can store all the parents in a single word.

    Edit Culling Lists
        - We can only process up to 256 edits in a single compute dispatch.
        - The edit storage is the edits uniform.
        - For each node of the octree, there is a array of 64 u32 indices, that references the 256 edits.
        - Each item in the list is used in intervals of 8 bits, so in one u32 word we store 4 indices.
        - With bit operations we can circunvent the lack of u8 of current webgpu, withou having to use the full u32 for indexing.
        - There are two structures:
            - The Culling Lists: where the actual lists are stored as a single buffer, accessed as a 2D array:
                    edit_culling_lists[in_list_word_index + octree_index * word_list_size] <- word_list_size is 64 here
            - Culling count: a buffer with the same strucutre as the octree, that stores the number of edits in each node

    Inside bricks:
        - For a correct representation, we would need to keep bricks inside the scuplture, in order for the substraction to work.
        - However, this seemed like a wastefull memory usage.
        - In order to prevent this, we found that marking in the octree the leaves that are inside the SDF as INTERIOR_BRICKS, and
          when substracting and needing to fill the inside bricks, we just set the default distance of 1000.0 of the default SDF to
          -1000.0 according to the flag.
        - This proved sufficient for this usecase. 

    Interval Arithmetic
        - In order to subdivide the octree and cull the edits in an accurate manner, we use a different version of the SDF function,
          that are built using interval arithmetic. You input a range in the function and they return the lowest and the biggest values
          that the function (in this case, the distance) can have in that interval.
        - We can use this in order to determine if there is surface in a particular area of our evaluation area: If the lowest value is less
          than 0, we know that in this area the distance is negative, and that means that there is AT LEAST a bit of surface in that block; 
          and we can subdivide in that area.
        
    SDF Exactness & boundness:
        - However when evaluating we found issues with SDF exactness. Some SDF are exact, and some operations also returns an exact SDF.
        - But, for example, Substraction results in a bound SDF, which means in practice that the propagation of the change in the distance
          is not correct; being only correct near where the operation or de SDF was placed.
        - This is fine for rendering, since the distances are "more correct" the closer you are, but in subdivision, this leads to incorrect
          culling. This is because due the nature of interval arithmetic, having an "incorrectly propagated" distance does not really change
          the minum and maximun interval in an area; unless it is a really big operation, slightly less bigger than the evaluationg surface.
        - In order to counter-meassure this, we subdivide using the incomming edits as exact Union operations, whose distances propagate 
          correctly (called new_edits).
        - Subdivision happens only on the area that is inside or along the surface of the incoming edits.
        - On the last layer, where we decide which brick do we write, delete or re-write, we decide based on the type of edit:
            - If its an union, we delete all bricks iside the resulting SDF, end write/re-evaluate the bricks on the surface of the resulting SDF.
            - If its a substraction, we delete all bricks inside the new_edits, and only create/re-evaluate bricks on the intersection of the 
              new_edits and the resulting SDF.

    TODO:
     - Redo edit culling with interval arithmetic.
     - Redo preview
     - Re-test reevaluation
     - Add back the constants for the ifs
     - Interval lack of precission
    // TODO(Juan): fix the preview

*/

fn is_inside_AABB(point : vec3f, aabb_min : vec3f, aabb_max : vec3f) -> bool {
    return (aabb_min.x <= point.x && point.x <= aabb_max.x) && (aabb_min.y <= point.y && point.y <= aabb_max.y) && (aabb_min.z <= point.z && point.z <= aabb_max.z);
}

fn fully_inside_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return is_inside_AABB(b1_min, b2_min, b2_max) && is_inside_AABB(b1_max, b2_min, b2_max);
}

fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
}

fn brick_remove(octree_index : u32) {
    // If its inside the new_edits, and the brick is filled, we delete it
    let brick_to_delete_idx = atomicAdd(&brick_buffers.brick_removal_counter, 1u);
    let instance_index : u32 = octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK;
    brick_buffers.brick_removal_buffer[brick_to_delete_idx] = instance_index;
    brick_buffers.brick_instance_data[instance_index].in_use = 0u;
    octree.data[octree_index].tile_pointer = 0u;
    octree.data[octree_index].octant_center_distance = vec2f(10000.0, 10000.0);
}

// Brick managing functions
fn brick_remove_and_mark_as_inside(octree_index : u32, is_current_brick_filled : bool) {
    if (is_current_brick_filled) {
        brick_remove(octree_index);
    } 
    octree.data[octree_index].tile_pointer = INTERIOR_BRICK_FLAG;
    octree.data[octree_index].octant_center_distance = vec2f(-10000.0, -10000.0);
}

fn brick_create_or_reevaluate(octree_index : u32, is_current_brick_filled : bool, is_interior_brick : bool, octant_center : vec3f) {
    let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 1);

    if (!is_current_brick_filled) {
        // Last level, more coarse culling
        let brick_spot_id = atomicSub(&brick_buffers.atlas_empty_bricks_counter, 1u) - 1u;
        let instance_index : u32 = brick_buffers.atlas_empty_bricks_buffer[brick_spot_id];
        brick_buffers.brick_instance_data[instance_index].position = octant_center;
        brick_buffers.brick_instance_data[instance_index].atlas_tile_index = instance_index;
        brick_buffers.brick_instance_data[instance_index].octree_parent_id = octree_index;
        brick_buffers.brick_instance_data[instance_index].in_use = BRICK_IN_USE_FLAG;


            octree.data[octree_index].tile_pointer = instance_index | FILLED_BRICK_FLAG;
    }
                
    octant_usage_write[prev_counter] = octree_index;
}

fn brick_reevaluate(octree_index : u32) {
    let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 1);
    octant_usage_write[prev_counter] = octree_index;
}

fn preview_brick_create(octree_index : u32, octant_center : vec3f, is_interior_brick : bool, edit_start_index : u32, edit_count : u32) {
    let preview_brick : u32 = atomicAdd(&brick_buffers.preview_instance_counter, 1u);
    
    brick_buffers.preview_instance_data[preview_brick].position = octant_center;
    brick_buffers.preview_instance_data[preview_brick].octree_parent_id = octree_index;
    brick_buffers.preview_instance_data[preview_brick].in_use = 0u;
    brick_buffers.preview_instance_data[preview_brick].edit_id_start = edit_start_index;
    brick_buffers.preview_instance_data[preview_brick].edit_count = edit_count;
    if (is_interior_brick) {
        brick_buffers.preview_instance_data[preview_brick].in_use = INTERIOR_BRICK_FLAG; 
    }
}

fn brick_mark_as_preview(octree_index : u32, edit_start_index : u32, edit_count : u32) {
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use |= BRICK_HAS_PREVIEW_FLAG;
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_id_start = edit_start_index;
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_count = edit_count;
}

fn brick_mark_as_hidden(octree_index : u32) {
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use |= BRICK_HIDE_FLAG;
}

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let level : u32 = atomicLoad(&octree.current_level);

    let id : u32 = group_id.x;

    var parent_level : u32;

    // Edge case: the first level does not have a parent, so it writes the edit list to itself
    if (level == 0) {
        parent_level = level;
    } else {
        parent_level = level - 1;
    }

    let octant_id : u32 = octant_usage_read[id];
    let parent_mask : u32 = u32(pow(2, f32(OCTREE_DEPTH * 3))) - 1;
    // The parent is indicated in the index, so according to the level, we remove the 3 lower bits, associated to the current octant
    let parent_octant_id : u32 = octant_id & (parent_mask >> (3u * (OCTREE_DEPTH - parent_level)));

    // In array indexing: in_level_position_of_octant (the octant id) + layer_start_in_array
    // Given a level, we can compute the size of a level with (8^(level-1))/7
    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
    let parent_octree_index : u32 = parent_octant_id + u32((pow(8.0, f32(parent_level)) - 1) / 7);

    // Culling list indices
    let curr_culling_layer_index = (octant_id + stroke_culling[0] * BRICK_COUNT) * stroke_history.count;
    let prev_culling_layer_index = (octant_id + ((stroke_culling[0] + 1u) % 2) * BRICK_COUNT) * stroke_history.count;

    var octant_center : vec3f = vec3f(0.0);
    var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

    //TODO(Juan): this could be LUT, but is it worthy at the expense of BIG SHADER
    // Compute the center and the half size of the current octree, in the current level, via iterating the octree index
    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));

        // For each level, select the octant position via the 3 corresponding bits and use the OCTREE_CHILD_OFFSET_LUT that
        // indicates the relative position of an octant in a layer
        // We offset the octant id depending on the layer that we are, and remove all the trailing bits (if any)
        octant_center += level_half_size * OCTREE_CHILD_OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x7];
    }

    // Note: the preview evaluation only happens at the end of the frame, so it must wait for
    //       any reevaluation and evaluation
    // TODO(Juan): fix undo redo reeval
    let is_evaluating_preview : bool = ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG);
    let is_evaluating_undo : bool = (octree.evaluation_mode & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;

    let octant_min : vec3f = octant_center - vec3f(level_half_size);
    let octant_max : vec3f = octant_center + vec3f(level_half_size);

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = vec2f(10000.0, 10000.0);
    var edit_counter : u32 = 0;


    // Base evaluation range
    let x_range : vec2f = vec2f(octant_center.x - level_half_size, octant_center.x + level_half_size);
    let y_range : vec2f = vec2f(octant_center.y - level_half_size, octant_center.y + level_half_size);
    let z_range : vec2f = vec2f(octant_center.z - level_half_size, octant_center.z + level_half_size);
    let eval_aabb_min : vec3f = vec3f(octant_center - level_half_size);
    let eval_aabb_max : vec3f = vec3f(octant_center + level_half_size);

    let current_subdivision_interval = iavec3_vecs(x_range, y_range, z_range);

    // For adition you can just use the intervals stored on the octree
    // however, for smooth substraction there can be precision issues
    // in the form of some bricks disappearing, and that can be solved by
    // recomputing the context

    var subdivide : bool = false;
    var margin : vec4f = vec4f(0.0);

    let is_current_brick_filled : bool = (octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG;
    let is_interior_brick : bool = (octree.data[octree_index].tile_pointer & INTERIOR_BRICK_FLAG) == INTERIOR_BRICK_FLAG;

    // NOTE: Code duplication due WGSL's rigid Pointer use & constraints
    if (!is_evaluating_preview) {
        // =====================================================
        // =====================================================
        //    _____ _             _          ______          _ 
        //   / ____| |           | |        |  ____|        | |
        //  | (___ | |_ _ __ ___ | | _____  | |____   ____ _| |
        //   \___ \| __| '__/ _ \| |/ / _ \ |  __\ \ / / _` | |
        //   ____) | |_| | | (_) |   <  __/ | |___\ V / (_| | |
        //  |_____/ \__|_|  \___/|_|\_\___| |______\_/ \__,_|_|
        // =====================================================
        // =====================================================
                                                    
        if (level == OCTREE_DEPTH) {
            // Compute the context of the current stroke,
            for (var j : u32 = 0; j < stroke_history.count; j++) {
                surface_interval = evaluate_stroke_interval(current_subdivision_interval, &(stroke_history.strokes[j]), &edit_list, surface_interval, octant_center, level_half_size);
            }
        
            // Pseudo subdivide!
            // Re-compute the strokes for the octants of the last level, and check the interval on those
            // Since the interval are smaller, the wrapping effect is lessend, and you add a brick if
            // at least one of this interval subdivisions is true.

            subdivide = true;
            // for (var i : u32 = 0; i < 8 && !subdivide; i++) {
            //     let sub_octant_id = octant_id | (i << (3 * level));

            //     let sub_level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32((level+1) + 1));

            //     let sub_octant_center = octant_center + sub_level_half_size * OCTREE_CHILD_OFFSET_LUT[(sub_octant_id >> (3 * ((level+1) - 1))) & 0x7];

            //     let x_range : vec2f = vec2f(sub_octant_center.x - sub_level_half_size, sub_octant_center.x + sub_level_half_size);
            //     let y_range : vec2f = vec2f(sub_octant_center.y - sub_level_half_size, sub_octant_center.y + sub_level_half_size);
            //     let z_range : vec2f = vec2f(sub_octant_center.z - sub_level_half_size, sub_octant_center.z + sub_level_half_size);
            //     let current_sub_interval = iavec3_vecs(x_range, y_range, z_range);

            //     var surf_interval : vec2f = vec2f(10000.0, 10000.0);
            //     for (var j : u32 = 0; j < stroke_history.count; j++) {
            //         surf_interval = evaluate_stroke_interval(current_sub_interval, &(stroke_history.strokes[j]), &edit_list, surf_interval, octant_center, level_half_size);
            //     }
                
            //     subdivide = surf_interval.x <= 0.0 && surf_interval.y >= 0.0;
            // }
        }

        // Do not evaluate all the bricks, only the ones whose distance interval has changed
        let prev_interval = octree.data[octree_index].octant_center_distance;
        octree.data[octree_index].octant_center_distance = surface_interval;
        
        if (level < OCTREE_DEPTH) {
            subdivide = intersection_AABB_AABB(eval_aabb_min, eval_aabb_max, stroke_history.eval_aabb_min, stroke_history.eval_aabb_max);
            
            if (subdivide) {
                // Stroke history culling
                var curr_stroke_count : u32 = 0u;
                for(var i : u32 = 0u; i < octree.data[parent_octree_index].stroke_count; i++) {
                    let index : u32 = culling_get_stroke_index(stroke_culling[prev_culling_layer_index + i + 1u]);
                    if (intersection_AABB_AABB(eval_aabb_min, 
                                               eval_aabb_max, 
                                               stroke_history.strokes[index].aabb_min, 
                                               stroke_history.strokes[index].aabb_max)) {
                        // Added to the current list
                        stroke_culling[curr_culling_layer_index + curr_stroke_count + 1u] = culling_get_culling_data(index, 0u, 0u);
                        curr_stroke_count = curr_stroke_count + 1u;
                    }
                }
                octree.data[octree_index].stroke_count = curr_stroke_count;

                // Subdivide
                // Increase the number of children from the current level
                let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 8);

                // Add to the index the childres's octant id, and save it for the next pass
                for (var i : u32 = 0; i < 8; i++) {
                    octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
                }
            }
        } else if (subdivide) {
            // in order to detect where the smooth factor is influencing  with the "goops"
            // We compare the two intervals,
            // in order to find the goops (where the current stroke is taking affect)

            let int_distance = abs(distance(prev_interval, surface_interval));
            
            if (int_distance > 0.00001) {
                if (surface_interval.x > 0.0) {
                    if (is_current_brick_filled) {
                        // delete any brick outside surface that was previosly filled
                        brick_remove(octree_index);
                    } else {
                        // reset flags for potential interior bricks
                        octree.data[octree_index].tile_pointer = 0;
                    }
                } else if (surface_interval.y < 0.0) {
                    brick_remove_and_mark_as_inside(octree_index, is_current_brick_filled);
                } else if (surface_interval.x < 0.0) {
                    brick_create_or_reevaluate(octree_index, is_current_brick_filled, is_interior_brick, octant_center);
                    // Stroke history culling
                    var curr_stroke_count : u32 = 0u;
                    for(var i : u32 = 0u; i < octree.data[parent_octree_index].stroke_count; i++) {
                        let index : u32 = culling_get_stroke_index(stroke_culling[prev_culling_layer_index + i + 1u]);
                        if (intersection_AABB_AABB(eval_aabb_min, 
                                                eval_aabb_max, 
                                                stroke_history.strokes[index].aabb_min, 
                                                stroke_history.strokes[index].aabb_max)) {
                            // Added to the current list
                            stroke_culling[curr_culling_layer_index + curr_stroke_count + 1u] = culling_get_culling_data(index, 0u, 0u);
                            curr_stroke_count = curr_stroke_count + 1u;
                        }
                    }
                    octree.data[octree_index].stroke_count = 1u;
                    octree.data[octree_index].culling_id = curr_culling_layer_index + 1u;
                }
            }
        } else if (is_current_brick_filled) {
            brick_remove(octree_index);
        }
    } else {
        // ============================================================
        // ============================================================
        //   _____                _                 ______          _   
        // |  __ \              (_)               |  ____|        | |  
        // | |__) | __ _____   ___  _____      __ | |____   ____ _| |  
        // |  ___/ '__/ _ \ \ / / |/ _ \ \ /\ / / |  __\ \ / / _` | |  
        // | |   | | |  __/\ V /| |  __/\ V  V /  | |___\ V / (_| | |_ 
        // |_|   |_|  \___| \_/ |_|\___| \_/\_/   |______\_/ \__,_|_(_)
        // =============================================================
        // =============================================================

        let SMOOTH_FACTOR : f32 = preview_stroke.stroke.parameters.w;
        
        if (level < OCTREE_DEPTH) {
            // Broad culling using only the incomming stroke
            // TODO: intersection with current edit AABB?
            if (intersection_AABB_AABB(eval_aabb_min, eval_aabb_max, merge_data.reevaluation_AABB_min, merge_data.reevaluation_AABB_max)) {
                // Subdivide
                // Increase the number of children from the current level
                let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 8);

                // Add to the index the childres's octant id, and save it for the next pass
                for (var i : u32 = 0; i < 8; i++) {
                    octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
                }
            }
        } else {
            let prev_interval : vec2f = octree.data[octree_index].octant_center_distance;
            let surface_with_preview_interval : vec2f = evaluate_stroke_interval(current_subdivision_interval,  &(preview_stroke.stroke), &(preview_stroke.edit_list), prev_interval, octant_center, level_half_size);
            let int_distance = abs(distance(prev_interval, surface_with_preview_interval));

            let in_surface_with_preview : bool = surface_with_preview_interval.x < 0.0 && surface_with_preview_interval.y > 0.0;
            let outside_surface_with_preview : bool = surface_with_preview_interval.x > 0.0 && surface_with_preview_interval.y > 0.0;
            let fully_inside_surface : bool = prev_interval.x < 0.0 && prev_interval.y < 0.0;
            let in_surface : bool = prev_interval.x < 0.0 && prev_interval.y > 0.0;
            let outside_surface : bool = prev_interval.x > 0.0 && prev_interval.y > 0.0;

            
            // if (int_distance > 0.00001) {
            //     if (surface_with_preview_interval.y < 0.0) {
            //         if (!is_current_brick_filled) {
            //             preview_brick_create(octree_index, octant_center, true);
            //         } else {
            //             brick_mark_as_preview(octree_index);
            //         }
            //     } else {
            //         brick_mark_as_preview(octree_index);
            //     }
            // } else {

            // }

            if (int_distance > 0.0001) {
                // Compute edit margin for preview evaluation
                var edit_index_start : u32 = 1000u;
                var edit_count : u32 = 0u;
                let starting_edit_pos : u32 = preview_stroke.stroke.edit_list_index;

                for(var i : u32 = 0u; i < preview_stroke.stroke.edit_count; i++) {
                    // WIP get AABB of current edit
                    let edit_pointer : ptr<storage, Edit> = &(preview_stroke.edit_list[i + starting_edit_pos]);

                    let half_size : vec3f = vec3f(edit_pointer.dimensions.x + preview_stroke.stroke.parameters.w);
                    let position : vec3f = (edit_pointer.position);

                    let aabb_min : vec3f = position - half_size;
                    let aabb_max : vec3f = position + half_size;

                    if (intersection_AABB_AABB(aabb_min, aabb_max, eval_aabb_min, eval_aabb_max)) {
                        edit_index_start = min(edit_index_start, i + starting_edit_pos);
                        edit_count = max(edit_count, i+1);
                    }
                }

                if (in_surface_with_preview) {

                    if (fully_inside_surface) {
                        preview_brick_create(octree_index, octant_center, true, edit_index_start, edit_count);
                    } else if (in_surface && is_current_brick_filled) {
                        brick_mark_as_preview(octree_index, edit_index_start, edit_count);
                    } else if (outside_surface) {
                        preview_brick_create(octree_index, octant_center, false, edit_index_start, edit_count);
                    } else if (in_surface) {
                        // preview_brick_create(octree_index, octant_center, true);
                    }
                } else if (outside_surface_with_preview && is_current_brick_filled) {
                    // TODO: hide this bricks!!
                    //brick_mark_as_preview(octree_index, edit_index_start, edit_count);
                    brick_mark_as_hidden(octree_index);
                }
            }
        }
    }
}