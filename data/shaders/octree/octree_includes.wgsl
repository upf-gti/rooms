#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define MAX_EDITS_PER_EVALUATION
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT
#define WORLD_SPACE_SCALE
#define STROKE_HISTORY_MAX_SIZE
#define BRICK_REMOVAL_COUNT

const SCULPT_TO_ATLAS_CONVERSION_FACTOR = (WORLD_SPACE_SCALE / SDF_RESOLUTION)  / (SCULPT_MAX_SIZE);
const PIXEL_WORLD_SIZE = SCULPT_MAX_SIZE / WORLD_SPACE_SCALE;
const PIXEL_ATLAS_SIZE = PIXEL_WORLD_SIZE * SCULPT_TO_ATLAS_CONVERSION_FACTOR;
const BRICK_WORLD_SIZE = 8.0 * PIXEL_WORLD_SIZE;
const BRICK_ATLAS_SIZE = 8.0 / SDF_RESOLUTION;
const PIXEL_WORLD_SIZE_QUARTER = PIXEL_WORLD_SIZE / 2;
const SQRT_3 = 1.73205080757;
const BRICK_COUNT = u32(SDF_RESOLUTION / 10.0);
const PACKED_LIST_SIZE : u32 = (MAX_EDITS_PER_EVALUATION / 4);
const TOTAL_BRICK_COUNT = BRICK_COUNT * BRICK_COUNT * BRICK_COUNT;

const OCTREE_TILE_INDEX_MASK = 0x3FFFFFFFu;

const FILLED_BRICK_FLAG = 0x80000000u;
const INTERIOR_BRICK_FLAG = 0x40000000u;

const UNDO_EVAL_FLAG = 0x01u;
const EVALUATE_PREVIEW_STROKE_FLAG = 0x02u;

const BRICK_IN_USE_FLAG = 0x001u;
const BRICK_HAS_PREVIEW_FLAG = 0x002u;
const BRICK_HIDE_FLAG = 0x004u;

const PREVIEW_BRICK_INSIDE_FLAG = 0x001u;

const OCTREE_CHILD_OFFSET_LUT : array<vec3f, 8> = array<vec3f, 8>(
    vec3f(-1.0, -1.0, -1.0),
    vec3f( 1.0, -1.0, -1.0),
    vec3f(-1.0,  1.0, -1.0),
    vec3f( 1.0,  1.0, -1.0),
    vec3f(-1.0, -1.0,  1.0),
    vec3f( 1.0, -1.0,  1.0),
    vec3f(-1.0,  1.0,  1.0),
    vec3f( 1.0,  1.0,  1.0)
);

struct Edit {
    position   : vec3f,
    dummy0     : f32,
    dimensions : vec4f,
    rotation   : vec4f,
    padding : vec4f
};

struct StrokeMaterial {
    roughness       : f32,
    metallic        : f32,
    emissive        : f32,
    dummy0          : f32,
    color           : vec4f,
    // Noise params
    noise_params    : vec4f,
    noise_color     : vec4f
};

//TODO(Juan): revisit the padding, and avoid using vec2/3/4 as padding
struct Stroke {
    stroke_id       : u32,
    edit_count      : u32,
    primitive       : u32,
    operation       : u32,
    parameters      : vec4f,
    edit_list_index : u32
    dummy1          : u32,
    dummy2          : u32,
    color_blend_op  : u32,
    dummy1          : vec4f,
    material        : StrokeMaterial   // 48 bytes,
};

struct StrokeHistory {
    count : u32,
    pad0:u32,
    pad1:u32,
    pad2:u32,
    eval_aabb_min : vec3f,
    pad3 : f32,
    eval_aabb_max : vec3f,
    pad4 : f32,
    pad5: vec4f,
    strokes : array<Stroke>
};

struct OctreeNode {
    octant_center_distance : vec2f,
    dummy : f32,
    tile_pointer : u32,
};

struct Octree {
    current_level : atomic<u32>,
    atomic_counter : atomic<u32>,
    proxy_instance_counter : atomic<u32>,
    evaluation_mode : u32,
    data : array<OctreeNode>
};

struct MergeData {
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
    padd : u32,
    padd2 : u32
};


struct IndirectBuffers {
    // For proxy brick dispatch
    brick_vertex_count : u32,
    brick_instance_count : atomic<u32>,
    brick_first_vertex : u32,
    brick_first_instance: u32,
     // For proxy preview dispatch
    preview_vertex_count : u32,
    preview_instance_count : atomic<u32>,
    preview_first_vertex : u32,
    preview_first_instance: u32,
    // For brick removal dispatch
    brick_removal_counter : atomic<u32>,
    brick_removal_padding1 : u32,
    brick_removal_padding2 : u32,
    brick_removal_padding3 : u32,
    // Evaluator subdivision indirect dispatch
    evaluator_subdivision_counter : u32,
    evaluator_subdivision_padding0 : u32,
    evaluator_subdivision_padding1 : u32,
    evaluator_subdivision_padding2 : u32
};

struct IndirectBuffers_ReadOnly {
    // For proxy brick dispatch
    brick_vertex_count : u32,
    brick_instance_count : u32,
    brick_first_vertex : u32,
    brick_first_instance: u32,
     // For proxy preview dispatch
    preview_vertex_count : u32,
    preview_instance_count : u32,
    preview_first_vertex : u32,
    preview_first_instance: u32,
    // For brick removal dispatch
    brick_removal_counter : u32,
    brick_removal_padding1 : u32,
    brick_removal_padding2 : u32,
    brick_removal_padding3 : u32,
    // Evaluator subdivision indirect dispatch
    evaluator_subdivision_counter : u32,
    evaluator_subdivision_padding0 : u32,
    evaluator_subdivision_padding1 : u32,
    evaluator_subdivision_padding2 : u32
};

struct BrickBuffers {
    atlas_empty_bricks_counter : atomic<u32>,
    brick_instance_counter : atomic<u32>,
    brick_removal_counter : atomic<u32>,
    preview_instance_counter : atomic<u32>,

    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,

    brick_removal_buffer : array<u32, BRICK_REMOVAL_COUNT>,

    brick_instance_data : array<ProxyInstanceData, BRICK_REMOVAL_COUNT>,

    preview_instance_data : array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct BrickBuffers_ReadOnly {
    atlas_empty_bricks_counter : u32,
    brick_instance_counter : u32,
    brick_removal_counter : u32,
    preview_instance_counter : u32,
    
    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,

    brick_removal_buffer : array<u32, BRICK_REMOVAL_COUNT>,

    brick_instance_data : array<ProxyInstanceData, BRICK_REMOVAL_COUNT>,

    preview_instance_data : array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct SculptData {
    sculpt_start_position   : vec3f,
    dummy1                  : f32,
    sculpt_rotation         : vec4f,
    sculpt_inv_rotation     : vec4f
};

struct RayIntersectionInfo
{
    intersected : u32,
    tile_pointer : u32,
    material_roughness : f32,
    material_metalness : f32,
    material_albedo : vec3f,
    dummy0 : u32,
    intersection_position : vec3f,
    dummy1 : u32,
};