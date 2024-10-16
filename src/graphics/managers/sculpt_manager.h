#pragma once

#include "framework/resources/sculpt.h"
#include "graphics/texture.h"


/*
    TODO:
        - Import mutiple sculpts
        - Delete sculpts
        - Raycasting
        - Compute preview
*/
class SculptManager {

    // GPU return data
    struct sGPU_Results {
        struct {
            uint32_t empty_brick_count = 0u;
            uint32_t sculpt_id = 0u;
            glm::vec3 aabb_min;
            glm::vec3 aabb_max;
        } sculpt_eval_data;

        struct {
            uint32_t    has_intersected = 0u;
            uint32_t    tile_pointer = 0u;
            uint32_t    sculpt_id = 0u;
            float       ray_t = 0.0;
        } ray_intersection;
    };

    WGPUBuffer      gpu_results_read_buffer = nullptr;
    Uniform         gpu_results_uniform;
    WGPUBindGroup   gpu_results_bindgroup = nullptr;

    // Sculpt instances
    struct sEvaluateRequest {
        Sculpt* sculpt = nullptr;
        sStrokeInfluence strokes_to_process; // context and new strokes in GPU format
        std::vector<Edit> edit_to_process; // context and new edits
    };

    bool previus_dispatch_had_preview = false;
    struct {
        Sculpt* sculpt = nullptr;
        bool needs_computing = false;
        sToUploadStroke to_upload_stroke;
        const std::vector<Edit>* to_upload_edit_list = nullptr;
    } preview;

    uint32_t        sculpt_count = 0u;

    bool performed_evaluation = false;

    std::vector<Sculpt*> sculpts_to_delete;
    std::vector<Sculpt*> sculpts_to_clean;
    std::vector<Sculpt*> sculpts_to_create;

    std::vector<sEvaluateRequest> evaluations_to_process;

    bool needs_evaluation = false;

    // Sculpt deletion
    Pipeline        sculpt_delete_pipeline;
    Shader*         sculpt_delete_shader = nullptr;
    WGPUBindGroup   compute_octree_clean_octree_bind_group = nullptr;

    // Sculpt ray intersection
    uint32_t intersections_to_compute = 0u;
    std::vector<Sculpt*> ray_intersection_to_compute;

    struct sGPU_RayData {
        glm::vec3   ray_origin;
        uint32_t    pad;
        glm::vec3   ray_direction;
    } ray_to_upload;

    struct sGPU_RayIntersection {
        uint32_t    has_intersected = 0u;
        uint32_t    tile_pointer = 0u;
        uint32_t    sculpt_id = 0u;
        float       ray_t = -FLT_MAX;
    };

    Uniform         ray_info_uniform;
    WGPUBindGroup   ray_info_bind_group = nullptr;

    Uniform         ray_intersection_info_uniform;
    WGPUBindGroup   ray_intersection_info_bind_group = nullptr;

    Pipeline        ray_intersection_pipeline;
    Shader*         ray_intersection_shader = nullptr;

    Pipeline        ray_intersection_result_and_clean_pipeline;
    Shader*         ray_intersection_result_and_clean_shader = nullptr;

    // Evaluation initialization
    Pipeline        evaluation_initialization_pipeline;
    Shader*         evaluation_initialization_shader = nullptr;
    WGPUBindGroup   evaluation_initialization_bind_group = nullptr;

    // Sculpt evaluation
    Pipeline evaluate_pipeline;
    Pipeline increment_level_pipeline;
    Pipeline write_to_texture_pipeline;
    Pipeline brick_removal_pipeline;
    Pipeline brick_copy_pipeline;
    Pipeline brick_unmark_pipeline;

    Shader* evaluate_shader = nullptr;
    Shader* increment_level_shader = nullptr;
    Shader* write_to_texture_shader = nullptr;
    Shader* brick_removal_shader = nullptr;
    Shader* brick_copy_shader = nullptr;
    Shader* brick_unmark_shader = nullptr;

    WGPUBindGroup   sdf_atlases_sampler_bindgroup = nullptr;
    WGPUBindGroup   evaluate_bind_group = nullptr;
    WGPUBindGroup   increment_level_bind_group = nullptr;
    WGPUBindGroup   write_to_texture_bind_group = nullptr;
    WGPUBindGroup   indirect_brick_removal_bind_group = nullptr;
    WGPUBindGroup   brick_copy_bind_group = nullptr;
    WGPUBindGroup   sculpt_delete_bindgroup = nullptr;
    WGPUBindGroup   brick_unmark_bind_group = nullptr;
    WGPUBindGroup   preview_stroke_bind_group = nullptr;

    // Evaluator uniforms
    uint32_t        octree_edit_list_size = 0u;
    Uniform         octree_edit_list;

    uint32_t        stroke_context_list_size = 0u;
    Uniform         stroke_context_list;

    Uniform         stroke_culling_buffer;

    Uniform         octant_usage_ping_pong_uniforms[4];
    Uniform         octant_usage_initialization_uniform[2];
    WGPUBindGroup   octant_usage_ping_pong_bind_groups[2] = {};

    // Preview evaluation


    // Octree struct reference
    struct sOctreeNode {
        glm::vec2   octant_center_distance = glm::vec2(10000.0f, 10000.0f);
        uint32_t    dummy = 0.0f;
        uint32_t    tile_pointer = 0;
        glm::vec3   padding;
        uint32_t    culling_data = 0u;
    };

    void clean_previous_preview(WGPUComputePassEncoder compute_pass);
    void upload_preview_strokes();
    void upload_strokes_and_edits(const std::vector<sToUploadStroke>& strokes_to_compute, const std::vector<Edit>& edits_to_upload);

    void init_shaders();
    void init_uniforms();
    void init_pipelines_and_bindgroups();

    void evaluate(WGPUComputePassEncoder compute_pass, const sEvaluateRequest& evaluate_request);
    void evaluate_preview(WGPUComputePassEncoder compute_pass);

    void evaluate_closest_ray_intersection(WGPUComputePassEncoder compute_pass);

public:
    void init();
    void clean();

    // TODO cleaning and deleting in frames
    void update(WGPUCommandEncoder command_encoder);

    void update_sculpt(Sculpt* sculpt, const sStrokeInfluence& strokes_to_process, const std::vector<Edit>& edits_to_process);
    void set_preview_stroke(Sculpt* sculpt, sToUploadStroke preview_stroke, const std::vector<Edit>& preview_edits);

    void test_ray_sculpts_intersection(const glm::vec3& ray_origin, const glm::vec3& ray_dir, const std::vector<Sculpt*> sculpts);

    Sculpt* create_sculpt();
    Sculpt* create_sculpt_from_history(const std::vector<Stroke>& stroke_history);

    bool has_performed_evaluation() { return performed_evaluation; }

    void delete_sculpt(WGPUComputePassEncoder compute_pass, Sculpt* to_delete);
};
