#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define MAX_EDITS_PER_EVALUATION
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT

const WORLD_SPACE_SCALE = 512.0; // A texel brick is 1/512 m
const SCALE_CONVERSION_FACTOR = WORLD_SPACE_SCALE / SDF_RESOLUTION;
const PIXEL_WORLD_SIZE = SCULPT_MAX_SIZE / WORLD_SPACE_SCALE;
const BRICK_WORLD_SIZE = 8.0 * PIXEL_WORLD_SIZE;
const SQRT_3 = 1.73205080757;
const BRICK_COUNT = u32(SDF_RESOLUTION / 10.0);
const PACKED_LIST_SIZE : u32 = (MAX_EDITS_PER_EVALUATION / 4);
const TOTAL_BRICK_COUNT = BRICK_COUNT * BRICK_COUNT * BRICK_COUNT;

const MIN_HIT_DIST = 0.00005;

const FILLED_BRICK_FLAG = 0x80000000u;
const INTERIOR_BRICK_FLAG = 0x40000000u;

const STROKE_CLEAN_BEFORE_EVAL_FLAG = 0x01u;
const EVALUATE_PREVIEW_STROKE_FLAG = 0x02u;

const BRICK_IN_USE_FLAG = 0x001u;
const BRICK_HAS_PREVIEW_FLAG = 0x002u;

const PREVIEW_BRICK_INSIDE_FLAG = 0x001u;

struct Edit {
    position   : vec3f,
    dummy0     : f32,
    dimensions : vec4f,
    rotation   : vec4f
};

struct Stroke {
    stroke_id       : u32,
    edit_count      : u32,
    primitive       : u32,
    operation       : u32,
    parameters      : vec4f,
    color           : vec4f,
    material        : vec4f,
    edits           : array<Edit, MAX_EDITS_PER_EVALUATION>
}

struct OctreeNode {
    octant_center_distance : vec2f,
    dummy : f32,
    tile_pointer : u32,
};

struct Octree {
    data : array<OctreeNode>
};

struct MergeData {
    sculpt_start_position : vec3f,
    max_octree_depth      : u32,
    sculpt_rotation       : vec4f,
    reevaluation_AABB_min : vec3f,
    reevaluate            : u32,
    reevaluation_AABB_max : vec3f,
    padding               : u32
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

struct PreviewProxyInstances {
    // Indirect buffer for dispatch
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    firt_instance: u32,
    // Instance data storage, for rendering
    instance_data: array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct PreviewProxyInstancesNonAtomic {
    // Indirect buffer for dispatch
    vertex_count : u32,
    instance_count : u32,
    first_vertex : u32,
    firt_instance: u32,
    // Instance data storage, for rendering
    instance_data: array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct OctreeProxyIndirect {
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    firt_instance: u32
};

struct OctreeState {
    current_level : atomic<u32>,
    atomic_counter : atomic<u32>,
    proxy_instance_counter : atomic<u32>,
    evaluation_mode : u32
};

struct EditCullingData {
    edit_culling_lists: array<u32, OCTREE_TOTAL_SIZE * MAX_EDITS_PER_EVALUATION / 4>,
    edit_culling_count : array<u32>
};