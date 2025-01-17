struct sEvaluatorDispatchCounter {
    wg_x : u32,
    wg_y : u32,
    wg_z : u32,
    //pad : u32
};

@group(0) @binding(0) var<storage, read_write> num_brinks_by_workgroup : u32;
@group(0) @binding(2) var<storage, read> bricks_to_write_to_tex_count : i32;
@group(0) @binding(3) var<storage, read> bricks_to_write_to_tex_buffer : array<u32>;
@group(0) @binding(4) var<storage, read_write> dispatch_counter : sEvaluatorDispatchCounter;

@compute @workgroup_size(1, 1, 1)
fn compute() 
{
    let t = bricks_to_write_to_tex_buffer[0];
    // I think the max is redundant...
    //num_brinks_by_workgroup = max(1u, u32(ceil(f32(bricks_to_write_to_tex_count) / 512.0)));
    let float_bricks_to_write_count : f32 = f32(bricks_to_write_to_tex_count);
    num_brinks_by_workgroup = u32(ceil(float_bricks_to_write_count / 65535.0));
    dispatch_counter.wg_x = u32(min(float_bricks_to_write_count, 65535.0));
}