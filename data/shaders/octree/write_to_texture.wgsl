#include ../sdf_functions.wgsl

struct MergeData {
    edits_aabb_start      : vec3<u32>,
    edits_to_process      : u32,
    sculpt_start_position : vec3f,
    max_octree_depth      : u32,
    sculpt_rotation       : vec4f
};

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

const SQRT_3 = 1.73205080757;

@compute @workgroup_size(8, 8, 8)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{

    let id : u32 = group_id.x;

    let octant_id : u32 = octant_usage_read[id];

    let level : u32 = atomicLoad(&current_level);

    // let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);

    var octant_center : vec3f = vec3f(0.0);

    var level_half_size : f32 = 0.5;

    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = 1.0 / pow(2.0, f32(i + 1));

        octant_center += level_half_size * OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x7];
    }

    // To start writing at the top corner
    let octant_corner : vec3f = octant_center - vec3f(level_half_size);

    let world_pos_texture_space : vec3f = octant_corner + vec3f(0.5);

    let start_writing_pos : vec3u = vec3u(world_pos_texture_space * SDF_RESOLUTION);

    var sSurface : Surface = Surface(vec3f(0.0, 0.0, 0.0), 10000.0);

    let pixel_offset : vec3f = vec3f(local_id) / SDF_RESOLUTION;

    for (var i : u32 = 0; i < merge_data.edits_to_process; i++) {

        var edit : Edit = edits.data[i];

        edit.position -= merge_data.sculpt_start_position;
        edit.position = rotate_point_quat(edit.position, merge_data.sculpt_rotation);

        edit.rotation = quat_mult(edit.rotation, quat_inverse(merge_data.sculpt_rotation));

        sSurface = evalEdit(octant_corner + pixel_offset, sSurface, edit);
    }

    // if (abs(sSurface.distance) < (level_half_size * SQRT_3)) {
        textureStore(write_sdf, start_writing_pos + local_id, vec4f(sSurface.color, sSurface.distance));
    // }

    octant_usage_write[0] = 0;
}
