#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT
#define WORLD_SPACE_SCALE
#define BRICK_REMOVAL_COUNT
#define MAX_SUBDIVISION_SIZE
#define MAX_STROKE_INFLUENCE_COUNT
#define OCTREE_LAST_LEVEL_STARTING_IDX

const SCULPT_TO_ATLAS_CONVERSION_FACTOR = (WORLD_SPACE_SCALE / SDF_RESOLUTION)  / (SCULPT_MAX_SIZE);
const PIXEL_WORLD_SIZE = SCULPT_MAX_SIZE / WORLD_SPACE_SCALE;
const PIXEL_ATLAS_SIZE = PIXEL_WORLD_SIZE * SCULPT_TO_ATLAS_CONVERSION_FACTOR;
const BRICK_WORLD_SIZE = 8.0 * PIXEL_WORLD_SIZE;
const BRICK_ATLAS_SIZE = 8.0 / SDF_RESOLUTION;
const PIXEL_WORLD_SIZE_QUARTER = PIXEL_WORLD_SIZE / 2;
const SQRT_3 = 1.73205080757;
const BRICK_COUNT = u32(SDF_RESOLUTION / 10.0);
const TOTAL_BRICK_COUNT = BRICK_COUNT * BRICK_COUNT * BRICK_COUNT;

const OCTREE_TILE_INDEX_MASK = 0x3FFFFFFFu;

const FILLED_BRICK_FLAG = 0x80000000u;
const INTERIOR_BRICK_FLAG = 0x40000000u;

const UNDO_EVAL_FLAG = 0x01u;
const EVALUATE_PREVIEW_STROKE_FLAG = 0x02u;

const BRICK_IN_USE_FLAG = 0x001u;
const BRICK_HAS_PREVIEW_FLAG = 0x002u;
const BRICK_HIDE_FLAG = 0x004u;

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

struct PreviewStroke {
    current_sculpt_idx : u32,
    dummy0    : u32,
    dummy1    : u32,
    dummy2    : u32,
    stroke    : Stroke,
    edit_list : array<Edit>
};

//TODO(Juan): revisit the padding, and avoid using vec2/3/4 as padding
struct Stroke {
    stroke_id       : u32,
    edit_count      : u32,
    primitive       : u32,
    operation       : u32,//4
    parameters      : vec4f,//4
    aabb_min        : vec3f,//4
    color_blend_op  : u32,
    aabb_max        : vec3f,
    edit_list_index : u32, //4
    material        : StrokeMaterial   // 48 bytes
};

struct StrokeHistory {
    count : u32,
    is_undo: u32,
    pad1: u32,
    pad2: u32,
    eval_aabb_min : vec3f,
    pad3 : f32,
    eval_aabb_max : vec3f,
    pad4 : f32,
    pad5: vec4f,
    strokes : array<Stroke>
};

struct OctreeNode {
    octant_center_distance : vec2f,
    stroke_count : u32,
    tile_pointer : u32,
    padding : vec3f,
    culling_id : u32
};

struct SculptIndirectCall {
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    first_instance : u32,
    brick_count : atomic<u32>,
    starting_model_idx : u32
};

struct SculptIndirectCall_NonAtomic {
    vertex_count : u32,
    instance_count : u32,
    first_vertex : u32,
    first_instance : u32,
    brick_count : u32,
    starting_model_idx : u32
};

struct Octree {
    current_level : atomic<u32>,
    atomic_counter : atomic<u32>,
    evaluation_mode : u32,
    octree_id : u32,
    data : array<OctreeNode>
};

struct Octree_NonAtomic {
    current_level : u32,
    atomic_counter : u32,
    evaluation_mode : u32,
    octree_id : u32,
    data : array<OctreeNode>
};

struct ProxyInstanceData {
    position : vec3f,
    atlas_tile_index : u32,
    octree_id : u32,
    in_use : u32,
    edit_id_start : u32,
    edit_count : u32
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
    brick_removal_counter : atomic<u32>,
    preview_instance_counter : atomic<u32>,

    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,

    brick_removal_buffer : array<u32, BRICK_REMOVAL_COUNT>,

    brick_instance_data : array<ProxyInstanceData, BRICK_REMOVAL_COUNT>,

    preview_instance_data : array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct BrickBuffers_ReadOnly {
    atlas_empty_bricks_counter : u32,
    brick_removal_counter : u32,
    preview_instance_counter : u32,
    
    atlas_empty_bricks_buffer : array<u32, TOTAL_BRICK_COUNT>,

    brick_removal_buffer : array<u32, BRICK_REMOVAL_COUNT>,

    brick_instance_data : array<ProxyInstanceData, BRICK_REMOVAL_COUNT>,

    preview_instance_data : array<ProxyInstanceData, PREVIEW_PROXY_BRICKS_COUNT>
};

struct GPUReturnResults {
    // Evaluation
    sculpt_aabb_min :      vec3f,
    empty_brick_count :    u32,
    sculpt_aabb_max :      vec3f,
    evaluation_sculpt_id : u32,

    // Ray interection
    ray_has_intersected :u32,
    ray_tile_pointer :      u32,
    ray_sculpt_id : u32,
    ray_t : f32,

    ray_sculpt_instance_id : u32,
    pad0 : u32,
    ray_metallic : f32,
    ray_roughness : f32,

    ray_albedo_color : vec3f,
    pad1 : u32
};

struct GPUReturnResults_Atomic {
    // Evaluation
    sculpt_aabb_min_x : atomic<i32>,
    sculpt_aabb_min_y : atomic<i32>,
    sculpt_aabb_min_z : atomic<i32>,
    empty_brick_count : u32,

    sculpt_aabb_max_x : atomic<i32>,
    sculpt_aabb_max_y : atomic<i32>,
    sculpt_aabb_max_z : atomic<i32>,
    evaluation_sculpt_id : u32,

    // Ray interection
    ray_has_intersected : u32,
    ray_tile_pointer : u32,
    ray_sculpt_id : u32,
    ray_t : f32,

    ray_sculpt_instance_id : u32,
    pad0 : u32,
    ray_metallic : f32,
    ray_roughness : f32,

    ray_albedo_color : vec3f,
    pad1 : u32
};

struct SculptInstanceData {
    flags       : u32,
    instance_id : u32,
    pad0        : u32,
    pad1        : u32,
    model       : mat4x4f,
    inv_model   : mat4x4f
};

const SCULPT_INSTANCE_NOT_SELECTED = 0u;
const SCULPT_INSTANCE_IS_OUT_OF_FOCUS = 1u;
const SCULPT_INSTANCE_IS_POINTED = 2u;
const SCULPT_INSTANCE_IS_SELECTED = 4u;

struct CullingStroke {
    stroke_idx : u32,
    edit_start_idx : u32,
    edit_count : u32
};
/**
    0-16 bits -> stroke id (0-65535 # of strokes)
    17-25 bits -> edit start (0-256)
    26-32 bits -> edit count (0-256)
*/
fn culling_stroke_get_edit_start_and_count(culling_data : u32, stroke_list : ptr<storage, array<Stroke>, read>) -> CullingStroke {
    let stroke_id : u32 = culling_data >> 16;
    let edit_start : u32 = (culling_data & 0xFF00u) >> 8;
    let edit_count : u32 = (culling_data & 0xFFu);

    let stroke_pointer : ptr<storage, Stroke, read> = &stroke_list[stroke_id];

    return CullingStroke(stroke_id, edit_start + stroke_pointer.edit_list_index, stroke_pointer.edit_count);
}

fn culling_get_culling_data(stroke_pointer : u32, edit_start : u32, edit_count : u32) -> u32 {
    var result : u32 = stroke_pointer << 16;
    // result |= (edit_start & 0xFFu) << 8;
    // result |= (edit_count & 0xFFu);

    return result;
}

fn culling_get_stroke_index(culling_data : u32) -> u32 {
    return (culling_data) >> 16;
}

struct sPaddedAABB {
    min    : vec3f,
    pad0        : u32,
    max    : vec3f,
    pad1        : u32
};