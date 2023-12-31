#pragma once

#include "includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "framework/aabb.h"

#include <list>

#define PREVIEW_EDITS_MAX 128
#define SDF_RESOLUTION 400
#define SCULPT_MAX_SIZE 1 // meters

class EntityMesh;

class RaymarchingRenderer {

    enum eEvaluatorOperationFlags : uint32_t {
        CLEAN_BEFORE_EVAL = 0x0001u,
        EVALUATE_PREVIEW_STROKE = 0x0002u
    };


    Uniform         linear_sampler_uniform;

    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Texture         sdf_texture;
    Uniform         sdf_texture_uniform;

    Texture         sdf_material_texture;
    Uniform         sdf_material_texture_uniform;

    // Octree creation
    Pipeline        compute_octree_evaluate_pipeline;
    Pipeline        compute_octree_increment_level_pipeline;
    Pipeline        compute_octree_write_to_texture_pipeline;
    Pipeline        compute_octree_brick_removal_pipeline;
    Pipeline        compute_octree_brick_copy_pipeline;
    Pipeline        compute_octree_initialization_pipeline;
    Pipeline        compute_octree_cleaning_pipeline;
    Shader*         compute_octree_evaluate_shader = nullptr;
    Shader*         compute_octree_increment_level_shader = nullptr;
    Shader*         compute_octree_write_to_texture_shader = nullptr;
    Shader*         compute_octree_brick_removal_shader = nullptr;
    Shader*         compute_octree_brick_copy_shader = nullptr;
    Shader*         compute_octree_initialization_shader = nullptr;
    Shader*         compute_octree_cleaning_shader = nullptr;
    WGPUBindGroup   compute_octree_evaluate_bind_group = nullptr;
    WGPUBindGroup   compute_octree_increment_level_bind_group = nullptr;
    WGPUBindGroup   compute_octree_write_to_texture_bind_group = nullptr;
    WGPUBindGroup   compute_octree_indirect_brick_removal_bind_group = nullptr;
    WGPUBindGroup   compute_octree_brick_copy_bind_group = nullptr;
    WGPUBindGroup   compute_octant_usage_bind_groups[2] = {};
    WGPUBindGroup   compute_stroke_buffer_bind_group = nullptr;
    WGPUBindGroup   compute_octree_initialization_bind_group = nullptr;
    WGPUBindGroup   compute_octree_clean_octree_bind_group = nullptr;

    Uniform         octree_uniform;
    Uniform         octant_usage_uniform[4];
    Uniform         octant_usage_initialization_uniform[2];
    uint8_t         octree_depth = 0;
    uint32_t        octants_max_size = 0;
    uint32_t        octree_total_size = 0;
    Uniform         octree_indirect_buffer;
    Uniform         octree_state;
    Uniform         octree_proxy_instance_buffer;
    Uniform         octree_proxy_indirect_buffer;
    Uniform         octree_edit_culling_data;
    Uniform         octree_indirect_brick_removal_buffer;
    Uniform         octree_brick_copy_buffer;
    Uniform         proxy_geometry_eye_position;
    WGPUBindGroup   render_camera_bind_group = nullptr;

    Uniform         sculpt_data_uniform;
    WGPUBindGroup   sculpt_data_bind_group = nullptr;

    Uniform         camera_uniform;

    Uniform         preview_stroke_uniform;

    Uniform         compute_merge_data_uniform;
    Uniform         compute_stroke_buffer_uniform;

    EntityMesh*     cube_mesh = nullptr;

    struct sSculptData {
        glm::vec3 sculpt_start_position = {0.f, 0.f, 0.f};
        float dummy1 = 0.0f;
        glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        glm::quat sculpt_inv_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    } sculpt_data;

    // Data needed for sdf merging
    struct sMergeData {
        glm::vec3  sculpt_start_position = { 0.f, 0.f, 0.f };
        uint32_t   max_octree_depth = 0;
        glm::quat  sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        glm::vec3  reevaluation_AABB_min;
        uint32_t   reevaluate = 0u;
        glm::vec3  reevaluation_AABB_max;
        uint32_t   padding;
    } compute_merge_data;

    struct sOctreeNode {
        glm::vec2 octant_center_distance = glm::vec2(10000.0f, 10000.0f);
        float dummy = 0.0f;
        uint32_t tile_pointer = 0;
    };

    Stroke current_stroke = {};
    Stroke in_frame_stroke = {};

    std::vector<Stroke> to_compute_stroke_buffer;
    std::vector<Stroke> stroke_history;
    std::list<Stroke> stroke_redo_history;

    // Preview edits
    //struct sPreviewEditsData {
    //    glm::vec3 aabb_center;
    //    float padding = 0.0f;
    //    glm::vec3 aabb_size;
    //    uint32_t preview_edits_count = 0u;
    //    Edit preview_edits[PREVIEW_EDITS_MAX];
    //} preview_edit_data;

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

    void evaluate_strokes(const std::vector<Stroke> strokes, bool is_undo = false, bool is_redo = false);

    void compute_preview_edit();

public:

    RaymarchingRenderer();

    int initialize(bool use_mirror_screen);
    void clean();

    void update(float delta_time);
    void render();

    void compute_octree();
    void render_raymarching_proxy(WGPURenderPassEncoder render_pass);

    void redo();
    void undo();

    void set_sculpt_start_position(const glm::vec3& position);
    void set_sculpt_rotation(const glm::quat& rotation);
    void set_camera_eye(const glm::vec3& eye_pos);

    /*
    *   Edits
    */
    void initialize_stroke();
    void change_stroke(const StrokeParameters& params, const uint32_t index_increment = 1u);

    void push_edit(const Edit edit);

    void push_edit_list(std::vector<Edit> &new_edits) {
        for (Edit &edit : new_edits) {
            push_edit(edit);
        }
    };

    void add_preview_edit(const Edit& edit);

    const glm::vec3& get_sculpt_start_position() { return compute_merge_data.sculpt_start_position; }
};
