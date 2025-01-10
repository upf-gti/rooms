#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
@group(0) @binding(7) var<storage, read> edit_list : array<Edit>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

@group(2) @binding(0) var<storage, read_write> bricks_to_write_to_tex_buffer : array<u32>;
@group(2) @binding(1) var<storage, read> bricks_to_eval_buffer : array<u32>;
@group(2) @binding(2) var<storage, read_write> bricks_to_eval_count : atomic<u32>;
@group(2) @binding(3) var<storage, read_write> bricks_to_write_to_tex_buffer_count : atomic<u32>;

#include sdf_interval_functions.wgsl

fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
}

// Brick managing functions
fn brick_remove(octree_index : u32) {
    // If its inside the new_edits, and the brick is filled, we delete it
    let brick_to_delete_idx = atomicAdd(&brick_buffers.brick_removal_counter, 1u);
    let instance_index : u32 = octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK;
    brick_buffers.brick_removal_buffer[brick_to_delete_idx] = instance_index;
    brick_buffers.brick_instance_data[instance_index].in_use = 0u;
    octree.data[octree_index].tile_pointer = 0u;
    octree.data[octree_index].octant_center_distance = vec2f(10000.0, 10000.0);
}

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
        brick_buffers.brick_instance_data[instance_index].in_use = BRICK_IN_USE_FLAG;
        brick_buffers.brick_instance_data[instance_index].octree_id = octree.octree_id;

        octree.data[octree_index].tile_pointer = instance_index | FILLED_BRICK_FLAG;
    }
                
    octant_usage_write[prev_counter] = octree_index;
}

fn brick_reevaluate(octree_index : u32)
{
    let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 1);
    octant_usage_write[prev_counter] = octree_index;
}

// fn preview_brick_create(octree_index : u32, octant_center : vec3f, is_interior_brick : bool, edit_start_index : u32, edit_count : u32)
// {
//     let preview_brick : u32 = atomicAdd(&brick_buffers.preview_instance_counter, 1u);
    
//     brick_buffers.preview_instance_data[preview_brick].position = octant_center;
//     brick_buffers.preview_instance_data[preview_brick].in_use = 0u;
//     brick_buffers.preview_instance_data[preview_brick].edit_id_start = edit_start_index;
//     brick_buffers.preview_instance_data[preview_brick].edit_count = edit_count;

//     if (is_interior_brick) {
//         brick_buffers.preview_instance_data[preview_brick].in_use = INTERIOR_BRICK_FLAG; 
//     }
// }

// fn brick_mark_as_preview(octree_index : u32, edit_start_index : u32, edit_count : u32)
// {
//     brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use |= BRICK_HAS_PREVIEW_FLAG;
//     brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_id_start = edit_start_index;
//     brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_count = edit_count;
// }

var<workgroup> brick_to_eval_wg_size : u32;
var<workgroup> bricks_to_eval_wg_buffer : array<u32, 512u>;

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let is_evaluating_undo : bool = (stroke_history.is_undo & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;
    let is_evaluating_undo_paint : bool = (stroke_history.is_undo & PAINT_UNDO_EVAL_FLAG) == PAINT_UNDO_EVAL_FLAG;

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = vec2f(10000.0, 10000.0);
    var edit_counter : u32 = 0;

    if (thread_id == 0u) {
        // Thread 0 of each workgroup handles the amount of work per thread
        let starting_brick_idx : i32 = atomicSub(&bricks_to_eval_count, 512);

        let work_count_tmp : i32 = starting_brick_idx - 512;
        // Check evaluator_culling_step for the math on this
        let work_count : u32 = u32(512 + (work_count_tmp) * i32(step(0.0, f32(work_count_tmp))));

        // Store the work count & bricks to the workgroup memory
        // is this needed?? maybe its not really neede due cache. TODO: test directly with VRAM
        brick_to_eval_wg_size = work_count;
        for(var i : u32 = 0u; i < work_count; i++) {
            bricks_to_eval_wg_buffer[i] = bricks_to_eval_buffer[starting_brick_idx - i];
        }  
    }

    workgroupBarrier();

    if (thread_id < brick_to_eval_wg_size) { 
        let octant_id : u32 = bricks_to_eval_wg_buffer[thread_id];

        let octree_index : u32 = current_brick_id + u32((pow(8.0, f32(OCTREE_DEPTH)) - 1) / 7);
        var octant_center : vec3f = vec3f(0.0);
        var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

        // Need a way to compute this WITHOUT the loop :(
        for (var i : u32 = 1; i <= OCTREE_DEPTH; i++) {
            // +1 is added to the pow exponent to get the half-size of current octant (otherwise would be size)
            level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));

            // For each level, select the octant position via the 3 corresponding bits and use the OCTREE_CHILD_OFFSET_LUT that
            // indicates the relative position of an octant in a layer
            // We offset the octant id depending on the layer that we are, and remove all the trailing bits (if any)
            octant_center += level_half_size * OCTREE_CHILD_OFFSET_LUT[(current_brick_id >> (3 * (i - 1))) & 0x7];
        }

        let eval_aabb_min : vec3f = octant_center - vec3f(level_half_size);
        let eval_aabb_max : vec3f = octant_center + vec3f(level_half_size);

        // Base evaluation range
        let x_range : vec2f = vec2f(octant_center.x - level_half_size, octant_center.x + level_half_size);
        let y_range : vec2f = vec2f(octant_center.y - level_half_size, octant_center.y + level_half_size);
        let z_range : vec2f = vec2f(octant_center.z - level_half_size, octant_center.z + level_half_size);
    
        let current_subdivision_interval = iavec3_vecs(x_range, y_range, z_range);

        let stroke_count : u32 = stroke_history.count;

        let culled_part : u32 = (min(stroke_history.count, MAX_STROKE_INFLUENCE_COUNT));
        let non_culled_count : u32 = ( (stroke_history.count) - culled_part);

        // Interval evaluation
        for(var i : u32 = 0u; i < stroke_count; i++) {
            if (stroke_history.strokes[i].operation != OP_SMOOTH_PAINT) {
                surface_interval = evaluate_stroke_interval(current_subdivision_interval, 
                                                            &(stroke_history.strokes[i]),
                                                            &edit_list, 
                                                            surface_interval, 
                                                            octant_center, 
                                                            level_half_size );
            } else {
                brick_has_paint = true;
            }
        }

        octree.data[octree_index].stroke_count = curr_stroke_count;
        octree.data[octree_index].culling_id = curr_culling_layer_index;

        // Do not evaluate all the bricks, only the ones whose distance interval has changed
        let prev_interval = octree.data[octree_index].octant_center_distance;
        octree.data[octree_index].octant_center_distance = surface_interval;

        let int_distance = abs(distance(prev_interval, surface_interval));
            
        if ((is_evaluating_undo_paint || brick_has_paint) && is_current_brick_filled) {
            brick_reevaluate(octree_index);
        } else if (int_distance > 0.0001) {
            if (surface_interval.x > 0.0) {
                if (is_current_brick_filled) {
                    // delete any brick outside surface that was previosly filled
                    brick_remove(octree_index);
                } else {
                    // // reset flags for potential interior bricks
                    octree.data[octree_index].tile_pointer = 0;
                    octree.data[octree_index].octant_center_distance = vec2f(10000.0, 10000.0);
                }
            } else if (surface_interval.y < 0.0) {
                brick_remove_and_mark_as_inside(octree_index, is_current_brick_filled);
            } else if (surface_interval.x < 0.0) {
                brick_create_or_reevaluate(octree_index, is_current_brick_filled, is_interior_brick, octant_center);
            }
        }
    }
}