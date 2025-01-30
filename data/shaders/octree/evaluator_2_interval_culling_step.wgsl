#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

struct sJobCounters {
    bricks_to_interval_eval_count : atomic<i32>,
    bricks_to_write_to_tex_count : atomic<i32>
};

@group(0) @binding(1) var<storage, read> bricks_to_interval_eval_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> job_counters : sJobCounters;
@group(0) @binding(3) var<storage, read_write> bricks_to_write_to_tex_buffer : array<u32>;

@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
@group(0) @binding(7) var<storage, read> edit_list : array<Edit>;
@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;


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

fn brick_create_or_reevaluate(octree_index : u32, is_current_brick_filled : bool, raw_brick_id_count : u32, is_interior_brick : bool, octant_center : vec3f) {
    let prev_counter : i32 = atomicAdd(&job_counters.bricks_to_write_to_tex_count, 1);

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
                
    bricks_to_write_to_tex_buffer[prev_counter] = raw_brick_id_count;
}

fn brick_reevaluate(octree_index : u32, raw_brick_id_count : u32)
{
    let prev_counter : i32 = atomicAdd(&job_counters.bricks_to_write_to_tex_count, 1);
    bricks_to_write_to_tex_buffer[prev_counter] = raw_brick_id_count;
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

// TODO: remove bricks is a bit funky

// NOTE: do not NAIVELY reduce thread group by doing more iterations per trehad
// brings perf down: occupancy might suffer. TODO: rethink how to reduce thread count
@compute @workgroup_size(8,8,8)
fn compute(@builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let is_evaluating_undo : bool = (stroke_history.is_undo & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;
    let is_evaluating_undo_paint : bool = (stroke_history.is_undo & PAINT_UNDO_EVAL_FLAG) == PAINT_UNDO_EVAL_FLAG;

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = vec2f(10000.0, 10000.0);
    var edit_counter : u32 = 0;

    if (thread_id == 0u) {
        // Thread 0 of each workgroup handles the amount of work per thread
        let starting_brick_idx : i32 = atomicSub(&job_counters.bricks_to_interval_eval_count, 512);

        //let work_count_tmp : i32 = starting_brick_idx - 512;
        // Check evaluator_culling_step for the math on this
        let work_count : i32 = clamp(starting_brick_idx, 0, 512);

        // Store the work count & bricks to the workgroup memory
        // is this needed?? maybe its not really neede due cache. TODO: test directly with VRAM
        brick_to_eval_wg_size = u32(work_count);
        for(var i : i32 = 0; i <= work_count; i++) {
            let raw_brick_id_count = bricks_to_interval_eval_buffer[starting_brick_idx - i - 1];
            bricks_to_eval_wg_buffer[i] = raw_brick_id_count;
        }  
    }

    workgroupBarrier();

    if (thread_id < brick_to_eval_wg_size) { 
        let raw_brick_id_count : u32 = bricks_to_eval_wg_buffer[thread_id];
        let brick_id : u32 = raw_brick_id_count >> 8u;
        let in_stroke_brick_count : u32 = raw_brick_id_count & 0xFF;
        let octree_index : u32 = brick_id + OCTREE_LAST_LEVEL_STARTING_IDX;

        var brick_center : vec3f = get_brick_center(brick_id);
        var level_half_size : f32 = 0.5 * BRICK_WORLD_SIZE;

        // Culling list indices
        let curr_culling_layer_index = brick_id * MAX_STROKE_INFLUENCE_COUNT;

        let eval_aabb_min : vec3f = brick_center - vec3f(level_half_size);
        let eval_aabb_max : vec3f = brick_center + vec3f(level_half_size);

        // Base evaluation range
        let x_range : vec2f = vec2f(brick_center.x - level_half_size, brick_center.x + level_half_size);
        let y_range : vec2f = vec2f(brick_center.y - level_half_size, brick_center.y + level_half_size);
        let z_range : vec2f = vec2f(brick_center.z - level_half_size, brick_center.z + level_half_size);
    
        let current_subdivision_interval = iavec3_vecs(x_range, y_range, z_range);

        let culled_part : u32 = min(stroke_history.count, MAX_STROKE_INFLUENCE_COUNT);
        let non_culled_count : u32 = stroke_history.count - culled_part;

        let is_current_brick_filled : bool = (octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG;
        let is_interior_brick : bool = (octree.data[octree_index].tile_pointer & INTERIOR_BRICK_FLAG) == INTERIOR_BRICK_FLAG;

        var brick_has_paint : bool = false;

        // Interval evaluation
        // Culled part
        for(var i : u32 = 0u; i < in_stroke_brick_count; i++) {
            let culled_idx : u32 = stroke_culling[curr_culling_layer_index + i];
            if (stroke_history.strokes[culled_idx].operation != OP_SMOOTH_PAINT) {
                surface_interval = evaluate_stroke_interval(current_subdivision_interval, 
                                                            &(stroke_history.strokes[culled_idx]),
                                                            &edit_list, 
                                                            surface_interval, 
                                                            brick_center, 
                                                            level_half_size );
            } else {
                brick_has_paint = true;
            }
        }

        // Non-culled part
        for(var i : u32 = 0u; i < non_culled_count; i++) {
            let non_culled_idx : u32 = i + MAX_STROKE_INFLUENCE_COUNT;
            if (stroke_history.strokes[non_culled_idx].operation != OP_SMOOTH_PAINT) {
                surface_interval = evaluate_stroke_interval(current_subdivision_interval, 
                                                            &(stroke_history.strokes[non_culled_idx]),
                                                            &edit_list, 
                                                            surface_interval, 
                                                            brick_center, 
                                                            level_half_size );
            } else {
                brick_has_paint = true;
            }
        }

        // TODO: re-do culling stuff <- Seguir por aquiiii
        //octree.data[octree_index].stroke_count = curr_stroke_count;
        //octree.data[octree_index].culling_id = curr_culling_layer_index;

        // Do not evaluate all the bricks, only the ones whose distance interval has changed
        let prev_interval = octree.data[octree_index].octant_center_distance;
        octree.data[octree_index].octant_center_distance = surface_interval;

        let int_distance = abs(distance(prev_interval, surface_interval));

        if ((is_evaluating_undo_paint || brick_has_paint) && is_current_brick_filled) {
            brick_reevaluate(octree_index, raw_brick_id_count);
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
                brick_create_or_reevaluate(octree_index, is_current_brick_filled, raw_brick_id_count, is_interior_brick, brick_center);
            }
        }
    }
}