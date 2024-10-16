#pragma once

#include "includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "framework/nodes/sculpt_instance.h"
#include "framework/math/aabb.h"

#include "stroke_manager.h"

#include <list>

#define OCTREE_DEPTH 6
#define BRICK_SIZE 10u
#define SSAA_SDF_WRITE_TO_TEXTURE false
#define PREVIEW_EDITS_MAX 128
#define SDF_RESOLUTION 400
#define SCULPT_MAX_SIZE 1 // meters
#define PREVIEW_PROXY_BRICKS_COUNT 7000u

#define MIN_SMOOTH_FACTOR 0.0001f
#define MAX_SMOOTH_FACTOR 0.02f

#define MIN_PRIMITIVE_SIZE 0.005f
#define MAX_PRIMITIVE_SIZE 0.08f

#define PREVIEW_BASE_EDIT_LIST 200u
#define PREVIEW_EDIT_LIST_INCREMENT 200u

class MeshInstance3D;

struct RayIntersectionInfo {
    uint32_t    intersected = 0;
    uint32_t    tile_pointer = 0;
    float       material_roughness = 0.0f;
    float       material_metalness = 0.0f;
    glm::vec3   material_albedo;
    uint32_t    dummy1 = 0;
    glm::vec3   intersection_position;
    uint32_t    dummy2 = 0;
};

class RaymarchingRenderer {

    enum eEvaluatorOperationFlags : uint32_t {
        CLEAN_BEFORE_EVAL = 0x0001u,
        EVALUATE_PREVIEW_STROKE = 0x0002u
    };

    uint32_t sculpt_count = 0u;
    std::vector<uint32_t> sculpt_instance_count;
    std::vector<SculptInstance*> sculpt_instances_list;

    std::vector<GPUSculptData> sculpts_to_delete;
    std::vector<GPUSculptData> sculpts_to_clean;
    std::vector<SculptInstance*> sculpts_to_process;

    Uniform         linear_sampler_uniform;

    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Pipeline        render_preview_proxy_geometry_pipeline;
    Shader*         render_preview_proxy_shader = nullptr;
    WGPUBindGroup   render_preview_proxy_geometry_bind_group = nullptr;
    WGPUBindGroup   render_preview_camera_bind_group = nullptr;

    Texture         sdf_texture;
    Uniform         sdf_texture_uniform;

    Texture         sdf_material_texture;
    Uniform         sdf_material_texture_uniform;

    // Octree parameters
    uint32_t        last_octree_level_size = 0u;
    uint32_t        max_brick_count = 0u;
    uint32_t        empty_brick_and_removal_buffer_count = 0u;
    float           brick_world_size = 0.0f;

    struct sBrickBuffers_counters {
        uint32_t atlas_empty_bricks_counter;
        uint32_t brick_instance_counter;
        uint32_t brick_removal_counter;
        uint32_t preview_instance_counter;
    };

    WGPUBuffer     brick_buffers_counters_read_buffer = nullptr;

    Uniform* sculpt_octree_uniform = nullptr;
    WGPUBindGroup sculpt_octree_bindgroup = nullptr;

    // Octree creation
    Pipeline        compute_octree_evaluate_pipeline;
    Pipeline        compute_octree_increment_level_pipeline;
    Pipeline        compute_octree_write_to_texture_pipeline;
    Pipeline        compute_octree_brick_removal_pipeline;
    Pipeline        compute_octree_brick_copy_pipeline;
    Pipeline        compute_octree_initialization_pipeline;
    Pipeline        compute_octree_cleaning_pipeline;
    Pipeline        compute_octree_ray_intersection_pipeline;
    Pipeline        compute_octree_brick_unmark_pipeline;
    Pipeline        sculpt_delete_pipeline;
    Shader*         compute_octree_evaluate_shader = nullptr;
    Shader*         compute_octree_increment_level_shader = nullptr;
    Shader*         compute_octree_write_to_texture_shader = nullptr;
    Shader*         compute_octree_brick_removal_shader = nullptr;
    Shader*         compute_octree_brick_copy_shader = nullptr;
    Shader*         compute_octree_initialization_shader = nullptr;
    Shader*         compute_octree_cleaning_shader = nullptr;
    Shader*         compute_octree_ray_intersection_shader = nullptr;
    Shader*         compute_octree_brick_unmark_shader = nullptr;
    Shader*         sculpt_delete_shader = nullptr;
    WGPUBindGroup   compute_octree_evaluate_bind_group = nullptr;
    WGPUBindGroup   compute_octree_increment_level_bind_group = nullptr;
    WGPUBindGroup   compute_octree_write_to_texture_bind_group = nullptr;
    WGPUBindGroup   compute_octree_indirect_brick_removal_bind_group = nullptr;
    WGPUBindGroup   compute_octree_brick_copy_bind_group = nullptr;
    WGPUBindGroup   compute_octant_usage_bind_groups[2] = {};
    WGPUBindGroup   compute_stroke_buffer_bind_group = nullptr;
    WGPUBindGroup   compute_octree_initialization_bind_group = nullptr;
    WGPUBindGroup   compute_octree_clean_octree_bind_group = nullptr;
    WGPUBindGroup   compute_octree_brick_unmark_bind_group = nullptr;
    WGPUBindGroup   brick_buffer_bindgroup = nullptr;

    Uniform         octant_usage_uniform[4];
    Uniform         octant_usage_initialization_uniform[2];
    uint8_t         octree_depth = 0;
    uint32_t        octants_max_size = 0;
    uint32_t        octree_total_size = 0;
    Uniform         octree_indirect_buffer_struct;
    Uniform         octree_indirect_buffer_struct_2;
    Uniform         octree_state;
    Uniform         octree_brick_buffers;
    Uniform         octree_preview_stroke;
    uint32_t        stroke_context_size = 0u;
    Uniform         octree_stroke_context;
    uint32_t        octree_edit_list_size;
    Uniform         octree_edit_list;
    Uniform         octree_brick_copy_buffer;
    WGPUBindGroup   render_camera_bind_group = nullptr;

    // Stroke culling data
    Uniform         stroke_culling_data;
    uint32_t        max_stroke_influence_count = 100u;

    Uniform         ray_info_uniform;
    Uniform         ray_intersection_info_uniform;
    WGPUBindGroup   octree_ray_intersection_bind_group = nullptr;
    WGPUBindGroup   octree_ray_intersection_info_bind_group = nullptr;
    WGPUBuffer      ray_intersection_info_read_buffer = nullptr;

    //Uniform         sculpt_data_uniform;
    Uniform         prev_stroke_uniform_2;
    WGPUBindGroup   sculpt_data_bind_proxy_group = nullptr;
    WGPUBindGroup   sculpt_data_bind_preview_group = nullptr;

    Uniform         *camera_uniform;

    Uniform         preview_stroke_uniform;
    WGPUBindGroup   preview_stroke_bind_group = nullptr;

    Uniform         compute_merge_data_uniform;
    Uniform         compute_stroke_buffer_uniform;

    Uniform         sculpt_model_buffer_uniform;

    Uniform         sculpt_instances_buffer_uniform;
    WGPUBindGroup   sculpt_instances_bindgroup = nullptr;

    MeshInstance3D* cube_mesh = nullptr;

    //struct sSculptData {
    //    glm::vec3 sculpt_start_position = {0.f, 0.f, 0.f};
    //    float dummy1 = 0.0f;
    //    glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    //    glm::quat sculpt_inv_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    //} sculpt_data;

    // Data needed for sdf merging
    struct sMergeData {
        glm::vec3  reevaluation_AABB_min;
        uint32_t   reevaluate = 0u;
        glm::vec3  reevaluation_AABB_max;
        uint32_t   padding;
    } compute_merge_data;

    struct sOctreeNode {
        glm::vec2 octant_center_distance = glm::vec2(10000.0f, 10000.0f);
        uint32_t dummy = 0.0f;
        uint32_t tile_pointer = 0;
        glm::vec3 padding;
        uint32_t culling_data = 0u;
    };

    struct RayInfo {
        glm::vec3 ray_origin;
        float dummy0;
        glm::vec3 ray_dir;
        float dummy1;
    } ray_info;

    RayIntersectionInfo ray_intersection_info;

    StrokeManager   stroke_manager = {};

    uint32_t preview_edit_array_length = 0u;
    struct sPreviewStroke {
        uint32_t current_sculpt_idx;
        uint32_t dummy0;
        uint32_t dummy1;
        uint32_t dummy2;
        sToUploadStroke stroke;
        std::vector<Edit> edit_list;

        AABB get_AABB() const;
    } preview_stroke;

    std::vector<Edit> incoming_edits;

    struct ProxyInstanceData {
        glm::vec3 position;
        uint32_t atlas_index;
        uint32_t octree_parent_index;
        uint32_t padding[3];
    };

    // Timestepping counters
    float updated_time = 0.0f;

    void init_compute_octree_pipeline();
    void init_raymarching_proxy_pipeline();
    void init_octree_ray_intersection_pipeline();

    void upload_stroke_context_data(sToComputeStrokeData* stroke_to_compute);

    void compute_delete_sculpts(WGPUComputePassEncoder compute_pass, GPUSculptData& to_delete);

    void evaluate_strokes(WGPUComputePassEncoder compute_pass);

    void compute_preview_edit(WGPUComputePassEncoder compute_pass);

    bool needs_undo = false;
    bool needs_redo = false;

    bool performed_evaluation = false;

    // DEBUG
    MeshInstance3D *AABB_mesh;

public:

    RaymarchingRenderer();

    int initialize(bool use_mirror_screen);
    void clean();

    void update_sculpt(WGPUCommandEncoder command_encoder);

    void compute_octree(WGPUCommandEncoder command_encoder, bool show_previews = false);
    void render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride = 0);

    inline void redo() {
        needs_redo = true;
    }
    inline void undo() {
        needs_undo = true;
    }

    bool has_performed_evaluation() { return performed_evaluation; }

    void set_current_sculpt(SculptInstance* sculpt_instance);
    SculptInstance* get_current_sculpt();

    void octree_ray_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, std::function<void(glm::vec3)> callback = nullptr);

    void get_brick_usage(std::function<void(float, uint32_t)> callback);

    const RayIntersectionInfo& get_ray_intersection_info() const;

    /*
    *   Edits
    */

    void initialize_stroke();
    void change_stroke(const StrokeParameters& params, const uint32_t index_increment = 1u);

    void push_edit(const Edit edit);

    void push_edit_list(std::vector<Edit> &new_edits) {
        for (const Edit &edit : new_edits) {
            push_edit(edit);
        }
    }

    inline void add_preview_edit(const Edit& edit) {
        if (preview_stroke.stroke.edit_count == preview_stroke.edit_list.size()) {
            preview_stroke.edit_list.resize(preview_stroke.edit_list.size() + PREVIEW_EDIT_LIST_INCREMENT);
        }
        preview_stroke.edit_list[preview_stroke.stroke.edit_count++] = edit;
    }


    void add_sculpt_instance(SculptInstance* instance);
    void remove_sculpt_instance(SculptInstance* instance);

    GPUSculptData create_new_sculpt();

    void create_sculpt_from_history(SculptInstance* instance, std::vector<Stroke>& stroke_history);
};
