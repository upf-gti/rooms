#include ../sdf_functions.wgsl

struct OctreeNode {
    tile_pointer : u32
}

struct Octree {
    data : array<OctreeNode>
};

struct MergeData {
    edits_aabb_start      : vec3<u32>,
    edits_to_process      : u32,
    sculpt_start_position : vec3f,
    dummy0                : f32,
    sculpt_rotation       : vec4f
};

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var<storage, read_write> atomic_counter : atomic<u32>;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

const SQRT_3 = 1.73205080757;

const OFFSET_LUT : array<vec3f, 8> = array<vec3f, 8>(
    vec3f(-1.0, -1.0, -1.0),
    vec3f( 1.0, -1.0, -1.0),
    vec3f(-1.0,  1.0, -1.0),
    vec3f( 1.0,  1.0, -1.0),
    vec3f(-1.0, -1.0,  1.0),
    vec3f( 1.0, -1.0,  1.0),
    vec3f(-1.0,  1.0,  1.0),
    vec3f( 1.0,  1.0,  1.0)
);

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let id : u32 = global_id.x;

    let octant_id : u32 = octant_usage_read[id];

    let level : u32 = atomicLoad(&current_level);

    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);

    var world_pos : vec3f = vec3f(0.50);

    var level_half_size : f32 = 0.5;

    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = 1.0 / pow(2.0, f32(i + 1));

        world_pos += level_half_size * OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x3];
    }

    var sSurface : Surface = Surface(vec3f(0.0, 0.0, 0.0), 10000.0);

    for (var i : u32 = 0; i < merge_data.edits_to_process; i++) {

        var edit : Edit = edits.data[i];

        //edit.position -= merge_data.sculpt_start_position;
        edit.position = rotate_point_quat(edit.position, merge_data.sculpt_rotation);

        edit.rotation = quat_mult(edit.rotation, quat_inverse(merge_data.sculpt_rotation));

        sSurface = evalEdit(world_pos, sSurface, edit);
    }

    if (sSurface.distance < level_half_size * SQRT_3) {
        let prev_counter : u32 = atomicAdd(&atomic_counter, 8);

        for (var i : u32 = 0; i < 8; i++) {
            octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
        }

        octree.data[octree_index].tile_pointer = 1;
    } else {
        octree.data[octree_index].tile_pointer = 0;
    }
}
