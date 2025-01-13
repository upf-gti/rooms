struct sEvaluatorDispatchCounter {
    wg_x : u32,
    wg_y : u32,
    wg_z : u32,
    //pad : u32
};

@group(0) @binding(0) var<storage, read_write> job_result_bricks_to_eval_count : i32;
@group(0) @binding(1) var<storage, read_write> job_result_bricks_to_eval : array<u32>;
@group(0) @binding(2) var<storage, read_write> aabb_culling_count : i32;
@group(0) @binding(3) var<storage, read_write> dispatch_counter : sEvaluatorDispatchCounter;


@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(local_invocation_index) thread_id: u32, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let t = aabb_culling_count;
    let j = job_result_bricks_to_eval[0];
    let i = job_result_bricks_to_eval_count;

    dispatch_counter.wg_x = u32(ceil(f32(dispatch_counter.wg_x) / 512.0)); 
}