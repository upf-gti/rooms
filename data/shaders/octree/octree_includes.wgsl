const SCULPT_MAX_SIZE = 1.0; // meters
const SDF_RESOLUTION = 400.0;
const WORLD_SPACE_SCALE = 512.0; // A texel brick is 1/512 m
const SCALE_CONVERSION_FACTOR = WORLD_SPACE_SCALE / SDF_RESOLUTION;
const PIXEL_WORLD_SIZE = SCULPT_MAX_SIZE / WORLD_SPACE_SCALE;
const BRICK_WORLD_SIZE = 8.0 * PIXEL_WORLD_SIZE;
const SQRT_3 = 1.73205080757;
const BRICK_COUNT = u32(SDF_RESOLUTION / 10.0);
const PACKED_LIST_SIZE : u32 = (64 / 4);
const TOTAL_BRICK_COUNT = BRICK_COUNT * BRICK_COUNT * BRICK_COUNT;

const MIN_HIT_DIST = 0.00005;

const FILLED_BRICK_FLAG = 0x80000000u;
const INTERIOR_BRICK_FLAG = 0x40000000u;

const SMOOTH_FACTOR = 0.01;

struct Edit {
    position   : vec3f,
    color      : vec3f,
    dimensions : vec4f,
    rotation   : vec4f
};

struct Stroke {
    stroke_id   : u32,
    edit_count  : u32,
    primitive   : u32,
    operation   : u32,
    parameters  : vec4f,
    edits       : array<Edit, 64>
}

struct OctreeNode {
    octant_center_distance : vec2f,
    dummy : f32,
    tile_pointer : u32,
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
    in_use : u32,
    padding : vec2u
};

struct IndirectBrickRemoval {
    brick_removal_counter : atomic<u32>,
    indirect_padding : vec3<u32>,
    brick_removal_buffer : array<u32>
};

struct IndirectBrickRemoval_ReadOnly {
    brick_removal_counter : u32,
    indirect_padding : vec3<u32>,
    brick_removal_buffer : array<u32>
};

struct OctreeProxyInstances {
    atlas_empty_bricks_counter : atomic<u32>,
    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,
    instance_data: array<ProxyInstanceData>
};

struct OctreeProxyInstancesNonAtomic {
    atlas_empty_bricks_counter : u32,
    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,
    instance_data: array<ProxyInstanceData>
};

struct OctreeProxyIndirect {
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    firt_instance: u32
};

struct OctreeCounters {
    current_level : atomic<u32>,
    atomic_counter : atomic<u32>,
    proxy_instance_counter : atomic<u32>
};