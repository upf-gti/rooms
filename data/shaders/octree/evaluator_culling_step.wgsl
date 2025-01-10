#include math.wgsl
#include octree_includes.wgsl

@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
//@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;


@group(1) @binding(0) var<storage, read_write> job_result_bricks_to_eval_count : atomic<u32>;
@group(1) @binding(1) var<storage, read_write> job_result_bricks_to_eval : array<u32>;
@group(1) @binding(2) var<storage, read_write> aabb_culling_count : atomic<i32>;


//@group(3) @binding(0) var<storage, read> preview_stroke : PreviewStroke;

/**
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

var<workgroup> workgroup_job_starting_idx : i32;
var<workgroup> workgroup_job_count : i32;
var<workgroup> workgroup_empy_thread_count : u32;

//const MAX_IT_COUNT : u32 = 1u;

// TODO: Only do one atomicAdd to shared memory per workgroup, and only write to shared memory with the same thread

fn add_brick_to_next_job_queue(brick_id : u32) {
    let idx : u32 = atomicAdd(&job_result_bricks_to_eval_count, 1u);
    job_result_bricks_to_eval[idx] = brick_id;
}

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    // Compute intersections for the last level directly

    let is_evaluating_undo : bool = (stroke_history.is_undo & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;

    let stroke_count : u32 = stroke_history.count;
    let stroke_history_aabb_min : vec3f = stroke_history.eval_aabb_min;
    let stroke_history_aabb_max : vec3f = stroke_history.eval_aabb_max;

    //for(var i : u32 = 0u; i < MAX_IT_COUNT; i++) {
        if (thread_id == 0u) {
            let prev_work_count : i32 = atomicSub(&aabb_culling_count, 512);

            // Configure the work for the workgroup
            workgroup_job_starting_idx = prev_work_count;
        
            // Reduce the number of jobs in a thread if there is no more
            let work_count_tmp : i32 = prev_work_count - 512;
            // there needs to be a batter way of doing this: this is supposed to be the "remainder":
            //      2000 jobs - 500 workers -> 500 job execution
            //      200 jobs - 500 workers -> 200 jobs
            workgroup_job_count = 512 + (work_count_tmp) * i32(step(0.0, f32(work_count_tmp)));
        }

        workgroupBarrier();

        // If the job count is bigger than the thread ID, there is no work for this thread
        if (thread_id < u32(workgroup_job_count)) {
            let current_brick_id : u32 = u32(workgroup_job_starting_idx) + thread_id;

            // Get the octree_idx from the last layer id
            //let octree_index : u32 = current_brick_id + u32((pow(8.0, f32(OCTREE_DEPTH)) - 1) / 7);
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

            // Test if it intersect with the current history to eval
            if (intersection_AABB_AABB( eval_aabb_min, 
                                        eval_aabb_max, 
                                        stroke_history_aabb_min, 
                                        stroke_history_aabb_max )) {
                if (is_evaluating_undo) {
                    // Add to the next work queue and early out
                    add_brick_to_next_job_queue(current_brick_id);
                } else {
                    // TODO: No Stroke history culling yet, only no crash (at this stage)
                    var any_stroke_inside : bool = false;
                    for(var i : u32 = 0u; i < stroke_count; i++) {
                        if (intersection_AABB_AABB( eval_aabb_min, 
                                                    eval_aabb_max, 
                                                    stroke_history.strokes[i].aabb_min, 
                                                    stroke_history.strokes[i].aabb_max  )) {
                            // Added to the current list
                            //curr_stroke_count = curr_stroke_count + 1u;
                            any_stroke_inside = true;
                            break; // <- early out
                        }
                    }

                    if (any_stroke_inside) {
                        // Add to the work queue
                        add_brick_to_next_job_queue(current_brick_id);
                    }
                }
            } 
        }
    //}
}