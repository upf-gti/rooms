#include math.wgsl
#include octree_includes.wgsl

struct sEvaluatorDispatchCounter {
    wg_x : atomic<u32>,
    wg_y : u32,
    wg_z : u32,
    //pad : u32
};

struct sJobCounters {
    bricks_to_interval_eval_count : atomic<i32>,
    bricks_to_write_to_tex_count : atomic<i32>
};

@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;


@group(1) @binding(2) var<storage, read_write> job_counter : sJobCounters;
@group(1) @binding(1) var<storage, read_write> job_result_bricks_to_eval : array<u32>;
@group(1) @binding(0) var<storage, read_write> aabb_culling_count : i32;
@group(1) @binding(3) var<storage, read_write> dispatch_counter : sEvaluatorDispatchCounter;



//@group(3) @binding(0) var<storage, read> preview_stroke : PreviewStroke;

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

// TODO: Only do one atomicAdd to shared memory per workgroup, and only write to shared memory with the same thread

fn add_brick_to_next_job_queue(brick_id : u32) {
    let idx : i32 = atomicAdd(&job_counter.bricks_to_interval_eval_count, 1);
    atomicAdd(&dispatch_counter.wg_x, 1u);
    job_result_bricks_to_eval[idx] = brick_id;
}

// A trheadgrounp of 64 should be better for occupancy
@compute @workgroup_size(4,4,4)
fn compute(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    // Compute intersections for the last level directly
    let global_id : u32 = (64u * wg_id.x + thread_id) * 8u;

    let is_evaluating_undo : bool = (stroke_history.is_undo & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;

    let stroke_count : u32 = stroke_history.count;
    let stroke_history_aabb_min : vec3f = stroke_history.eval_aabb_min;
    let stroke_history_aabb_max : vec3f = stroke_history.eval_aabb_max;

    var in_brick_stroke_count : u32 = 0u;

    // Reduce the threadcount, but do more work per thread
    for(var j : u32 = 0u; j < 8u; j++) {
        let operation_id : u32 = global_id + j;
        // If the job count is bigger than the thread ID, there is no work for this thread
        if (operation_id < u32(aabb_culling_count)) {
            // Get the octree_idx from the last layer id
            var brick_center : vec3f = get_brick_center(operation_id);

            // Culling list indices
            let curr_culling_layer_index = operation_id * MAX_STROKE_INFLUENCE_COUNT;

            var brick_half_size : f32 = 0.5 * BRICK_WORLD_SIZE;

            let eval_aabb_min : vec3f = brick_center - vec3f(brick_half_size);
            let eval_aabb_max : vec3f = brick_center + vec3f(brick_half_size);

            // Test if it intersect with the current history to eval
            if (intersection_AABB_AABB( eval_aabb_min, 
                                        eval_aabb_max, 
                                        stroke_history_aabb_min, 
                                        stroke_history_aabb_max )) {
                // if (is_evaluating_undo) {
                //     // Add to the next work queue and early out
                //     add_brick_to_next_job_queue(operation_id);
                // } else {
                    // TODO: No Stroke history culling yet, only no crash (at this stage)
                    //var any_stroke_inside : bool = false;
                    for(var i : u32 = 0u; i < stroke_count; i++) {
                        if (intersection_AABB_AABB( eval_aabb_min, 
                                                    eval_aabb_max, 
                                                    stroke_history.strokes[i].aabb_min, 
                                                    stroke_history.strokes[i].aabb_max  )) {
                            // Added to the current list
                            //curr_stroke_count = curr_stroke_count + 1u;
                            if (in_brick_stroke_count < MAX_STROKE_INFLUENCE_COUNT) {
                                stroke_culling[curr_culling_layer_index + in_brick_stroke_count] = i;
                                in_brick_stroke_count = in_brick_stroke_count + 1u;
                            }
                            
                            //any_stroke_inside = true;
                            //break; // <- early out
                        }
                    }

                    if (in_brick_stroke_count > 0u || is_evaluating_undo) {
                        // Add to the work queue
                        add_brick_to_next_job_queue((operation_id << 8u) | in_brick_stroke_count);
                    }
                //}
            } 
        }
    }
}