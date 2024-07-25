#define SDF_RESOLUTION
#define SCULPT_MAX_SIZE
#define OCTREE_DEPTH
#define OCTREE_TOTAL_SIZE
#define PREVIEW_PROXY_BRICKS_COUNT
#define WORLD_SPACE_SCALE
#define BRICK_REMOVAL_COUNT
#define MAX_SUBDIVISION_SIZE
#define MAX_STROKE_INFLUENCE_COUNT

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

/**
    pkd_edit_count_params
        edit count -> 16 bits
        SDF parameters -> 5 + 5 + 6 = 16 bits
    pkd_ops_prim_blending
        ops -> 3 bits
        primitive - > 8 bits
        color blending -> 4 bits
*/
// struct Stroke {
//     start_edit_idx          : u32,
//     pkd_edit_count_params   : u32,
//     pkd_ops_prim_blending   : u32,
//     pkd_material            : u32
// };

// fn stroke_is_smooth_paint(stroke : ptr<Stroke, storage>) -> bool {
//     return (stroke.pkd_ops_prim_blending & 0x60000000u) == 0x60000000u;
// }

// fn stroke_get_edit_data(stroke : ptr<Stroke, storage>) -> vec2u {
//     return { stroke.start_edit_idx, (stroke.pkd_edit_count_params >> 16u)};
// }

// fn stroke_get_params(stroke : ptr<Stroke, storage>) -> vec3f {
//     let blend_raw : u32 = stroke.pkd_edit_count_params;
//     let x : f32 = f32((blend_raw & 0xF800u) >> 11u);
//     let y : f32 = f32((blend_raw & 0x7C0u) >> 6u);
//     let z : f32 = f32(blend_raw & 0x3Fu);

//     return vec3f(x / 32.0, y / 32.0, z / 64.0);
// }

// fn stroke_get_color_blending_option(stroke : ptr<Stroke, storage>) -> u32 {
//     return (stroke.pkd_ops_prim_blending & 0x1E0000u) >> 17u;
// }

// fn stroke_get_op_and_prim(stroke : ptr<Stroke, storage>) -> vec2u {
//     let ops_prim : u32 = stroke.pkd_ops_prim_blending;
//     return { (ops_prim & 0xE0000000u) >> 29u, (ops_prim & 0x1FE00000u) >> 21u};
// }

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