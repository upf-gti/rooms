
const SDF_RESOLUTION = 512.0;
const SQRT_3 = 1.73205080757;
const SCULPT_MAX_SIZE = 2.0; // meters
const BRICK_COUNT = u32(SDF_RESOLUTION / 10.0);
const PACKED_LIST_SIZE : u32 = (64 / 4);

struct Edit {
    position   : vec3f,
    primitive  : u32,
    color      : vec3f,
    operation  : u32,
    dimensions : vec4f,
    rotation   : vec4f,
    parameters : vec4f
};

struct Edits {
    data : array<Edit, 64>
}

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
    atlas_tile_index : u32,
    octree_parent_id : u32, // a hack I dont like it
    padding : vec3u
};

struct OctreeCounters {
    current_level : atomic<u32>,
    atomic_counter : atomic<u32>,
    atlas_tile_counter : atomic<u32>
};