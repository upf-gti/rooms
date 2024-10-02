#pragma once

#include "includes.h"
#include "rooms_includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "framework/nodes/sculpt_instance.h"
#include "framework/math/aabb.h"

#include "stroke_manager.h"

#include <list>

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

enum eSculptInstanceFlags : uint32_t {
    SCULPT_NOT_SELECTED = 0u,
    SCULPT_IS_OUT_OF_FOCUS = 0b1u,
    SCULPT_IS_POINTED = 0b10u,
    SCULPT_IS_SELECTED = 0b100u
};

struct sSculptInstanceData {
    uint32_t flags = 0u;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    glm::mat4x4 model;
};

class RaymarchingRenderer {

    enum eEvaluatorOperationFlags : uint32_t {
        CLEAN_BEFORE_EVAL = 0x0001u,
        EVALUATE_PREVIEW_STROKE = 0x0002u
    };

    uint32_t sculpt_count = 0u;
    std::vector<uint32_t> sculpt_instance_count;
    std::vector<SculptInstance*> sculpt_instances_list;

    std::vector<SculptInstance*> sculpts_to_process;

    Uniform         linear_sampler_uniform;

    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Pipeline        render_preview_proxy_geometry_pipeline;
    Shader*         render_preview_proxy_shader = nullptr;
    WGPUBindGroup   render_preview_proxy_geometry_bind_group = nullptr;
    WGPUBindGroup   render_preview_camera_bind_group = nullptr;


    // Octree parameters
    

    struct sBrickBuffers_counters {
        uint32_t atlas_empty_bricks_counter;
        uint32_t brick_instance_counter;
        uint32_t brick_removal_counter;
        uint32_t preview_instance_counter;
    };

    //WGPUBuffer     brick_buffers_counters_read_buffer = nullptr;

    Uniform* sculpt_octree_uniform = nullptr;
    WGPUBindGroup sculpt_octree_bindgroup = nullptr;
    uint32_t current_sculpt_id;

    // Octree creation
    
    Pipeline        compute_octree_ray_intersection_pipeline;
    Pipeline        compute_octree_brick_unmark_pipeline;
    
    Shader*         compute_octree_ray_intersection_shader = nullptr;
    Shader*         compute_octree_brick_unmark_shader = nullptr;
    
    WGPUBindGroup   compute_stroke_buffer_bind_group = nullptr;
    WGPUBindGroup   compute_octree_brick_unmark_bind_group = nullptr;
    
    
    Uniform         octree_preview_stroke;
    
    WGPUBindGroup   render_camera_bind_group = nullptr;

    

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

    Uniform         compute_stroke_buffer_uniform;

    Uniform         sculpts_instance_data_uniform;

    Uniform         sculpt_instances_buffer_uniform;
    WGPUBindGroup   sculpt_instances_bindgroup = nullptr;

    MeshInstance3D* cube_mesh = nullptr;

    //struct sSculptData {
    //    glm::vec3 sculpt_start_position = {0.f, 0.f, 0.f};
    //    float dummy1 = 0.0f;
    //    glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    //    glm::quat sculpt_inv_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    //} sculpt_data;

    

    

    struct RayInfo {
        glm::vec3 ray_origin;
        float dummy0;
        glm::vec3 ray_dir;
        float dummy1;
    } ray_info;

    RayIntersectionInfo ray_intersection_info;

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

    // Timestepping counters
    float updated_time = 0.0f;

    void init_compute_octree_pipeline();
    void init_raymarching_proxy_pipeline();
    void init_octree_ray_intersection_pipeline();

    void upload_stroke_context_data(sToComputeStrokeData* stroke_to_compute);

    void compute_preview_edit(WGPUComputePassEncoder compute_pass);

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

    inline void add_preview_edit(const Edit& edit) {
        if (preview_stroke.stroke.edit_count == preview_stroke.edit_list.size()) {
            preview_stroke.edit_list.resize(preview_stroke.edit_list.size() + PREVIEW_EDIT_LIST_INCREMENT);
        }
        preview_stroke.edit_list[preview_stroke.stroke.edit_count++] = edit;
    }

    /*
    *   Sculpt management
    */

    void add_sculpt_instance(SculptInstance* instance);
    void remove_sculpt_instance(SculptInstance* instance);
};
