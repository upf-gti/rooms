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
@group(0) @binding(2) var<storage, read_write> sdf_data : SdfData;

fn evalSdf(position : vec3u) -> Surface
{
    var sSurface : Surface = sdf_data.data[position.x + position.y * 512 + position.z * 512 * 512];

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
    sdf_data.data[world_id.x + world_id.y * 512 + world_id.z * 512 * 512] = evalSdf(world_id);
}
