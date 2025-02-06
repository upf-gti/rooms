#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

struct sJobCounters {
    bricks_to_interval_eval_count : atomic<i32>,
    bricks_to_write_to_tex_count : atomic<i32>
};

@group(0) @binding(0) var<storage, read> preview_stroke : PreviewStroke;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

#include sdf_interval_functions.wgsl

/**
    NOTE: the biggest bottlenecks of the evaluator are 2 constnat costs:
    Culling step & brick copy, each takes like 0.3 ms, so its a fixed cost of 0.6 ms (ROUGHT!)
    The culling can be reduced if we change the dispatch size to match the aabb of the 
    stroke history (currently there are a lot of wasted work in here)
     The briuck_copy / AABB gen is much more difficult to optimize atm
    
    Since the number of bricks to test is always the same, we just use an atomic for the job
    counter, there is no need for a job queue:

        aabb_job_count = 2000

    For each workgroup:
        prev_aabb_job_count = atomicSub(aabb_job_count, 1000)

    Now prev_aabb_job_count is 2000, aabb_job_count is 1000
    With prev_aabb_job_count we know that the jobs for this workgroup is for bricks 2000 to 1000
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

fn preview_brick_create(octree_index : u32, octant_center : vec3f, is_interior_brick : bool, edit_start_index : u32, edit_count : u32)
{
    let preview_brick : u32 = atomicAdd(&brick_buffers.preview_instance_counter, 1u);
    
    brick_buffers.preview_instance_data[preview_brick].position = octant_center;
    brick_buffers.preview_instance_data[preview_brick].in_use = 0u;
    brick_buffers.preview_instance_data[preview_brick].edit_id_start = edit_start_index;
    brick_buffers.preview_instance_data[preview_brick].edit_count = edit_count;

    if (is_interior_brick) {
        brick_buffers.preview_instance_data[preview_brick].in_use = INTERIOR_BRICK_FLAG; 
    }

    atomicAdd(&indirect_buffers.preview_instance_count, 1u);
}

fn brick_unmark_preview(octree_index : u32) {
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use &= ~BRICK_HAS_PREVIEW_FLAG;
}

fn brick_mark_as_preview(octree_index : u32, edit_start_index : u32, edit_count : u32)
{
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use |= BRICK_HAS_PREVIEW_FLAG;
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_id_start = edit_start_index;
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].edit_count = edit_count;
}

fn brick_mark_as_hidden(octree_index : u32)
{
    brick_buffers.brick_instance_data[octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK].in_use |= BRICK_HIDE_FLAG;
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

// A trheadgrounp of 64 should be better for occupancy
@compute @workgroup_size(4,4,4)
fn compute(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    // Compute intersections for the last level directly
    let global_id : u32 = (64u * wg_id.x + thread_id) * 8u;

    // Reduce the threadcount, but do more work per thread
    for(var j : u32 = 0u; j < 8u; j++) {
        let operation_id : u32 = global_id + j;
        var in_brick_stroke_count : u32 = 0u;

        // If the job count is bigger than the thread ID, there is no work for this thread
        if (operation_id < MAX_SUBDIVISION_SIZE) {
            // Get the octree_idx from the last layer id
            var brick_center : vec3f = get_octant_center_at_level_wo_halfsize(operation_id, OCTREE_DEPTH);


            let octree_index : u32 = operation_id + OCTREE_LAST_LEVEL_STARTING_IDX;
            var brick_half_size : f32 = 0.5 * BRICK_WORLD_SIZE;

            let eval_aabb_min : vec3f = brick_center - vec3f(brick_half_size);
            let eval_aabb_max : vec3f = brick_center + vec3f(brick_half_size);

            // Base evaluation range
            let x_range : vec2f = vec2f(brick_center.x - brick_half_size, brick_center.x + brick_half_size);
            let y_range : vec2f = vec2f(brick_center.y - brick_half_size, brick_center.y + brick_half_size);
            let z_range : vec2f = vec2f(brick_center.z - brick_half_size, brick_center.z + brick_half_size);

            let current_subdivision_interval = iavec3_vecs(x_range, y_range, z_range);

            // Test if it intersect with the preview AABB
            if (intersection_AABB_AABB( eval_aabb_min, 
                                        eval_aabb_max, 
                                        preview_stroke.stroke.aabb_min, 
                                        preview_stroke.stroke.aabb_max)) {
                 let prev_interval : vec2f = octree.data[octree_index].octant_center_distance;
                let surface_with_preview_interval : vec2f = evaluate_stroke_interval(current_subdivision_interval,  &(preview_stroke.stroke), &(preview_stroke.edit_list), prev_interval, brick_center, brick_half_size);
                let int_distance = abs(distance(prev_interval, surface_with_preview_interval));

                let in_surface_with_preview : bool = surface_with_preview_interval.x < 0.0 && surface_with_preview_interval.y > 0.0;
                let outside_surface_with_preview : bool = surface_with_preview_interval.x > 0.0 && surface_with_preview_interval.y > 0.0;
                let fully_inside_surface : bool = prev_interval.x < 0.0 && prev_interval.y < 0.0;
                let in_surface : bool = prev_interval.x < 0.0 && prev_interval.y > 0.0;
                let outside_surface : bool = prev_interval.x > 0.0 && prev_interval.y > 0.0;

                let is_paint : bool = preview_stroke.stroke.operation == OP_SMOOTH_PAINT;

                if (int_distance > 0.0001 || is_paint) {
                    // Compute edit margin for preview evaluation
                    var edit_index_start : u32 = 10000u;
                    var edit_index_end : u32 = 0u;
                    let starting_edit_pos : u32 = preview_stroke.stroke.edit_list_index;

                    let preview_aabb_half_size : mat4x3f = get_loose_half_size_mat(preview_stroke.stroke.primitive);

                    let smooth_margin : vec3f = vec3f(preview_stroke.stroke.parameters.w);

                    let is_current_brick_filled : bool = (octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG;

                    for(var i : u32 = 0u; i < preview_stroke.stroke.edit_count; i++) {
                        // WIP get AABB of current edit
                        let current_idx : u32 = i + starting_edit_pos;
                        let edit_pointer : ptr<storage, Edit> = &(preview_stroke.edit_list[current_idx]);

                        let half_size : vec3f = (preview_aabb_half_size * edit_pointer.dimensions) + smooth_margin;
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
                            preview_brick_create(octree_index, brick_center, true, edit_index_start, edit_count);
                        } else if (in_surface && is_current_brick_filled) {
                            brick_mark_as_preview(octree_index, edit_index_start, edit_count);
                        } else if (outside_surface) {
                            preview_brick_create(octree_index, brick_center, false, edit_index_start, edit_count);
                        } else if (in_surface) {
                            // preview_brick_create(octree_index, brick_center, true);
                        }
                    } else if (outside_surface_with_preview && is_current_brick_filled) {
                        // TODO: restablish hidden functionality (when culling is implemented)
                        //brick_mark_as_hidden(octree_index);
                        brick_mark_as_preview(octree_index, edit_index_start, edit_count);
                    } else {
                        brick_unmark_preview(octree_index);
                    }
                } else {
                    brick_unmark_preview(octree_index);
                }
            } else {
                brick_unmark_preview(octree_index);
            }
        }
    }
}