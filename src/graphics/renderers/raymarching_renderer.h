#pragma once

#include "includes.h"
#include "rooms_includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "framework/nodes/sculpt_instance.h"
#include "framework/math/aabb.h"

#include "graphics/managers/stroke_manager.h"

#include <list>


/*
    TODO:
        - Make dynamic the instance data buffers
        - Preview
*/

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

#define MAX_INSTANCES_PER_SCULPT 20u

class RaymarchingRenderer {

    enum eEvaluatorOperationFlags : uint32_t {
        CLEAN_BEFORE_EVAL = 0x0001u,
        EVALUATE_PREVIEW_STROKE = 0x0002u
    };

    // Render pipelines
    struct sSculptInstanceData {
        uint32_t flags = 0u;
        uint32_t pad0;
        uint32_t pad1;
        uint32_t pad2;
        glm::mat4x4 model;
    };

    struct sSculptRenderInstances {
        Sculpt* sculpt = nullptr;
        uint16_t instance_count = 0u;
        sSculptInstanceData models[MAX_INSTANCES_PER_SCULPT];
    };

    std::unordered_map<uint32_t, sSculptRenderInstances*> sculpts_render_lists;
    std::vector<sSculptInstanceData> models_for_upload;
    Uniform         global_sculpts_instance_data_uniform;

    struct {
        uint32_t                count = 20u;
        std::vector<uint32_t>   count_buffer;
        Uniform                 uniform_count_buffer;
        WGPUBindGroup           count_bindgroup = nullptr;

        Pipeline                prepare_indirect;
        Shader*                 prepare_indirect_shader = nullptr;
    } sculpt_instances;

    Uniform         linear_sampler_uniform;

    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Pipeline        render_preview_proxy_geometry_pipeline;
    Shader*         render_preview_proxy_shader = nullptr;
    WGPUBindGroup   render_preview_proxy_geometry_bind_group = nullptr;
    WGPUBindGroup   render_preview_camera_bind_group = nullptr;

    /* Uniform         ray_info_uniform;
    Uniform         ray_intersection_info_uniform;
    WGPUBindGroup   octree_ray_intersection_bind_group = nullptr;
    WGPUBindGroup   octree_ray_intersection_info_bind_group = nullptr;
    WGPUBuffer      ray_intersection_info_read_buffer = nullptr;*/

    //Pipeline        compute_octree_ray_intersection_pipeline;
    //Shader*         compute_octree_ray_intersection_shader = nullptr;

    Pipeline        compute_octree_brick_unmark_pipeline;
    Shader*         compute_octree_brick_unmark_shader = nullptr;
    WGPUBindGroup   compute_octree_brick_unmark_bind_group = nullptr;

    WGPUBindGroup   sculpt_data_bind_preview_group = nullptr;

    Uniform*        camera_uniform;
    WGPUBindGroup   render_camera_bind_group = nullptr;

    Uniform         compute_stroke_buffer_uniform;

    MeshInstance3D* cube_mesh = nullptr;

    bool            render_preview = false;
 
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

    void init_raymarching_proxy_pipeline();
    void init_octree_ray_intersection_pipeline();

    void render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride = 0);

    void render_preview_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride = 0);


    // DEBUG
    MeshInstance3D *AABB_mesh;

public:

    RaymarchingRenderer();

    int initialize(bool use_mirror_screen);

    void clean();

    void add_rendercall_to_sculpt(Sculpt* sculpt, const glm::mat4& model, const uint32_t flags = 0u);

    void update_sculpts_and_instances(WGPUCommandEncoder command_encoder);

    void render(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride = 0u);

    void octree_ray_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, std::function<void(glm::vec3)> callback = nullptr);

    void get_brick_usage(std::function<void(float, uint32_t)> callback);

    const RayIntersectionInfo& get_ray_intersection_info() const;

    inline void set_preview_render(const bool need_to_render_preview) {
        render_preview = need_to_render_preview;
    }

    /*
    *   Edits
    */

    inline void add_preview_edit(const Edit& edit) {
        if (preview_stroke.stroke.edit_count == preview_stroke.edit_list.size()) {
            preview_stroke.edit_list.resize(preview_stroke.edit_list.size() + PREVIEW_EDIT_LIST_INCREMENT);
        }
        preview_stroke.edit_list[preview_stroke.stroke.edit_count++] = edit;
    }

    /*
    *   Sculpt management
    */

    //void add_sculpt_instance(SculptInstance* instance);
    //void remove_sculpt_instance(SculptInstance* instance);
};
