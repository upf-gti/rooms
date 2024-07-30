#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT
#define WORLD_SPACE_SCALE
#define BRICK_REMOVAL_COUNT
#define MAX_SUBDIVISION_SIZE
#define MAX_STROKE_INFLUENCE_COUNT

const HALF_MAX_STROKE_INFLUENCE_COUNT = (MAX_STROKE_INFLUENCE_COUNT / 2u);
const QUARTER_MAX_SUBDIVISION_SIZE = (MAX_SUBDIVISION_SIZE / 4u);

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

/**
    pkd_edit_count_params
        edit count -> 16 bits
        SDF parameters -> 5 + 5 + 6 = 16 bits
    pkd_ops_prim_blending
        ops -> 3 bits
        primitive - > 8 bits
        color blending -> 4 bits
*/
struct Stroke {
    start_edit_idx          : u32,
    pkd_edit_count_params   : u32,
    pkd_ops_prim_blending   : u32,
    pkd_material            : u32
};

fn stroke_is_smooth_paint(stroke : ptr<storage, Stroke>) -> bool {
    return (stroke.pkd_ops_prim_blending & 0x60000000u) == 0x60000000u;
}

fn stroke_get_edit_data(stroke : ptr<storage, Stroke>) -> vec2u {
    return vec2u(stroke.start_edit_idx, (stroke.pkd_edit_count_params >> 16u));
}

fn stroke_get_params(stroke : ptr<storage, Stroke>) -> vec4f {
    let blend_raw : u32 = stroke.pkd_edit_count_params;
    let x : f32 = f32((blend_raw & 0xF800u) >> 11u);
    let y : f32 = f32((blend_raw & 0x7C0u) >> 6u);
    let z : f32 = f32(blend_raw & 0x3Fu);

    return vec4f(x / 31.0, y / 31.0, z / 63.0, 0.0);
}

fn stroke_get_color_blending_option(stroke : ptr<storage, Stroke>) -> u32 {
    return (stroke.pkd_ops_prim_blending & 0x1E0000u) >> 17u;
}

fn stroke_get_op_and_prim(stroke : ptr<storage, Stroke>) -> vec2u {
    let ops_prim : u32 = stroke.pkd_ops_prim_blending;
    return vec2u((ops_prim & 0xE0000000u) >> 29u, (ops_prim & 0x1FE00000u) >> 21u);
}

struct AABB {
    min : vec3f,
    padd0 : u32,
    max : vec3f,
    padd1 : u32
};

struct AABB_List {
    stroke_count : u32,
    padd0 : u32,
    padd1 : u32,
    padd2 : u32,
    aabbs : array<AABB>
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
    stroke_count : u32,
    tile_pointer : u32,
    padding : vec3f,
    culling_id : u32
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

struct MergeData {
    evaluation_AABB_min : vec3f,
    reevaluate            : u32,
    evaluation_AABB_max : vec3f,
    padding               : u32
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

// struct SculptData {
//     sculpt_start_position   : vec3f,
//     dummy1                  : f32,
//     sculpt_rotation         : vec4f,
//     sculpt_inv_rotation     : vec4f
// };

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

const STROKE_INDEX_SIZE = MAX_STROKE_INFLUENCE_COUNT * QUARTER_MAX_SUBDIVISION_SIZE;
const STROKE_INDEX_HALF_SIZE = STROKE_INDEX_SIZE / 2u;
// NOTE: dont access the index buffer as is, since it is u16 packed into u32
struct StrokeCullingBuffers {
    stroke_index_count : atomic<u32>,
    padd0 : u32,
    padd1 : u32,
    padd2 : u32,
    stroke_index_buffer : array<u32, STROKE_INDEX_SIZE>,
    stroke_indices_per_node_buffer : array<u32, MAX_SUBDIVISION_SIZE * 2u>
};

struct StrokeCullingBuffers_NonAtomic {
    stroke_index_count : u32,
    padd0 : u32,
    padd1 : u32,
    padd2 : u32,
    stroke_index_buffer : array<u32, STROKE_INDEX_SIZE>,
    stroke_indices_per_node_buffer : array<u32, MAX_SUBDIVISION_SIZE * 2u>
};