#include sdf_functions.wgsl

struct MergeData {
    edits_aabb_start      : vec3<u32>,
    edits_to_process      : u32,
    sculpt_start_position : vec3f,
    dummy0                : f32,
    sculpt_rotation       : vec4f
};

struct SdfData {
    data : array<Surface>
};

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;

@group(0) @binding(2) var read_sdf: texture_3d<f32>;
@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba32float, write>;

fn evalSdf(position : vec3u) -> Surface
{
    var sdf_load : vec4f = textureLoad(read_sdf, position, 0);
    var sSurface : Surface = Surface(sdf_load.xyz, sdf_load.w);

    for (var i : u32 = 0; i < merge_data.edits_to_process; i++) {

        var edit : Edit = edits.data[i];

        edit.position -= merge_data.sculpt_start_position;
        edit.position = rotate_point_quat(edit.position, merge_data.sculpt_rotation);

        edit.rotation = quat_mult(edit.rotation, quat_inverse(merge_data.sculpt_rotation));

        sSurface = evalEdit(vec3f(position), sSurface, edit);
    }

    return sSurface;
}

@compute @workgroup_size(8, 8, 8)

fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let world_id : vec3<u32> = merge_data.edits_aabb_start + id;
    let result_sdf : Surface = evalSdf(world_id);
    textureStore(write_sdf, world_id, vec4f(result_sdf.color, result_sdf.distance));
}
