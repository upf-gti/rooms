
const SQRT_3 = 1.73205080757;
const SCULPT_MAX_SIZE = 2.0; // meters
const BRICK_COUNT = u32(512.0 / 10.0);

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
    max_octree_depth      : u32,
    sculpt_rotation       : vec4f
};

struct ProxyInstanceData {
    position : vec3f,
    atlas_tile_index : u32
};
