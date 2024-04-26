#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define MAX_EDITS_PER_EVALUATION
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT
#define WORLD_SPACE_SCALE
#define STROKE_HISTORY_MAX_SIZE

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

const STROKE_CLEAN_BEFORE_EVAL_FLAG = 0x01u;
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

struct Stroke {
    stroke_id       : u32,
    edit_count      : u32,
    primitive       : u32,
    operation       : u32,
    parameters      : vec4f,
    dummy           : vec3f,
    color_blend_op  : u32,
    dummy1          : vec4f,
    material        : StrokeMaterial,   // 48 bytes
    edits           : array<Edit, MAX_EDITS_PER_EVALUATION>
};

struct StrokeHistory {
    count : u32,
    pad0:u32,
    pad1:u32,
    pad2:u32,
    pad12: vec4f,
    pad22: vec4f,
    pad32: vec4f,
    strokes : array<Stroke, STROKE_HISTORY_MAX_SIZE>
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

struct PreviewData {
    // Indirect buffer for dispatch
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    firt_instance: u32,
    // The stroke for the preview
    preview_stroke : Stroke,
    // Instance data storage, for rendering
    instance_data: array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct PreviewDataReadonly {
    // Indirect buffer for dispatch
    vertex_count : u32,
    instance_count : u32,
    first_vertex : u32,
    firt_instance: u32,
    // The stroke for the preview
    preview_stroke : Stroke,
    // Instance data storage, for rendering
    instance_data: array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
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
    // Sculpt
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    firt_instance: u32
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