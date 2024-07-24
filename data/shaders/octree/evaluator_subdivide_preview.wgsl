#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

@group(0) @binding(1) var<uniform> merge_data : MergeData;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

@group(3) @binding(0) var<storage, read> preview_stroke : PreviewStroke;


#include sdf_interval_functions.wgsl

fn is_inside_AABB(point : vec3f, aabb_min : vec3f, aabb_max : vec3f) -> bool {
    return (aabb_min.x <= point.x && point.x <= aabb_max.x) && (aabb_min.y <= point.y && point.y <= aabb_max.y) && (aabb_min.z <= point.z && point.z <= aabb_max.z);
}

fn fully_inside_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return is_inside_AABB(b1_min, b2_min, b2_max) && is_inside_AABB(b1_max, b2_min, b2_max);
}

fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
}

fn get_loose_half_size_mat(prim : u32) -> mat4x3f
{
    if (prim == SD_SPHERE) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    } else if (prim == SD_BOX) {
        return mat4x3f(vec3f(1.0, 0.50, 0.50), vec3f(0.50, 1.0, 0.50), vec3f(0.50, 0.50, 1.0), vec3f(0.0));
    } else if (prim == SD_CAPSULE) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0), vec3f(0.0), vec3f(0.0));
    } else if (prim == SD_CONE) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(0.50, 1.0, 0.50), vec3f(0.0), vec3f(0.0));
    } else if (prim == SD_CYLINDER) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(0.50, 0.5, 0.50), vec3f(0.0), vec3f(0.0));
    } else if (prim == SD_TORUS) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0), vec3f(0.0), vec3f(0.0));
    } else if (prim == SD_VESICA) {
        return mat4x3f(vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.5, 0.0), vec3f(0.0), vec3f(1.0));
    }

    return mat4x3f();
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

    let eval_aabb_min : vec3f = vec3f(octant_center - level_half_size);
    let eval_aabb_max : vec3f = vec3f(octant_center + level_half_size);

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = vec2f(10000.0, 10000.0);
    var edit_counter : u32 = 0;

    // For adition you can just use the intervals stored on the octree
    // however, for smooth substraction there can be precision issues
    // in the form of some bricks disappearing, and that can be solved by
    // recomputing the context

    var subdivide : bool = false;
    var margin : vec4f = vec4f(0.0);

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

            let is_paint : bool = preview_stroke.stroke.operation == OP_SMOOTH_PAINT;

            if (int_distance > 0.0001 || is_paint) {
                // Compute edit margin for preview evaluation
                var edit_index_start : u32 = 1000u;
                var edit_index_end : u32 = 0u;
                let starting_edit_pos : u32 = preview_stroke.stroke.edit_list_index;

                let aabb_half_size : mat4x3f = get_loose_half_size_mat(preview_stroke.stroke.primitive);

                let smooth_margin : vec3f = vec3f(preview_stroke.stroke.parameters.w);

                for(var i : u32 = 0u; i < preview_stroke.stroke.edit_count; i++) {
                    // WIP get AABB of current edit
                    let current_idx : u32 = i + starting_edit_pos;
                    let edit_pointer : ptr<storage, Edit> = &(preview_stroke.edit_list[current_idx]);

                    let half_size : vec3f = (aabb_half_size * edit_pointer.dimensions) + smooth_margin;
                    let position : vec3f = (edit_pointer.position);

                    let aabb_min : vec3f = position - half_size;
                    let aabb_max : vec3f = position + half_size;

                    if (intersection_AABB_AABB(aabb_min, aabb_max, eval_aabb_min, eval_aabb_max)) {
                        edit_index_start = min(edit_index_start, current_idx);
                        edit_index_end = max(edit_index_end, current_idx + 1u);
                    }
                }

                let edit_count : u32 = edit_index_end - edit_index_start;

                if (is_paint && is_current_brick_filled) {
                    brick_mark_as_preview(octree_index, edit_index_start, edit_count);
                } else if (in_surface_with_preview) {
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
                    brick_mark_as_hidden(octree_index);
                }
            }
    }
}