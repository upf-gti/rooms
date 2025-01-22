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

struct sGPU_RayIntersectionData {
    uint32_t    has_intersected = 0u;
    uint32_t    tile_pointer = 0u;
    uint32_t    sculpt_id = 0u;
    float       ray_t = -FLT_MAX;

    uint32_t    instance_id = 0u;
    uint32_t    pad0;
    float       intersection_metallic;
    float       intersection_roughness;

    glm::vec3   intersection_albedo;
    uint32_t    pad1;
};

// GPU return data
struct sGPU_SculptResults {
    struct sGPU_SculptEvalData {
        glm::vec3 aabb_min = glm::vec3(6.0f);
        uint32_t empty_brick_count = 0u;

        glm::vec3 aabb_max = glm::vec3(3.0f);
        uint32_t sculpt_id = 0u;

        uint32_t curr_sculpt_brick_count = 0u;
        uint32_t pad0 = 0u;
        uint32_t pad1 = 0u;
        uint32_t pad2 = 0u;
    } sculpt_eval_data;

    sGPU_RayIntersectionData ray_intersection;
};

class SculptNode;

class SculptManager {

    Uniform         gpu_results_uniform;
    WGPUBindGroup   gpu_results_bindgroup = nullptr;

    // Sculpt instances
    struct sEvaluateRequest {
        Sculpt* sculpt = nullptr;
        sStrokeInfluence strokes_to_process; // context and new strokes in GPU format
        uint32_t edit_count = 0u;
        std::vector<Edit> edit_to_process; // context and new edits
    };

    bool previus_dispatch_had_preview = false;
    struct {
        uint32_t sculpt_model_idx = 0u;
        Sculpt* sculpt = nullptr;
        bool needs_computing = false;
        sGPUStroke to_upload_stroke;
        const std::vector<Edit>* to_upload_edit_list = nullptr;
    } preview;

    uint32_t        sculpt_count = 0u;

    bool performed_evaluation = false;

    std::vector<Sculpt*> sculpts_to_delete;
    std::vector<Sculpt*> sculpts_to_clean;

    std::vector<sEvaluateRequest> evaluations_to_process;

    bool needs_evaluation = false;

    // Sculpt deletion
    Pipeline        sculpt_delete_pipeline;
    Shader*         sculpt_delete_shader = nullptr;

    // Sculpt ray intersection
    uint32_t intersections_to_compute = 0u;
    SculptNode* intersection_node_to_test = nullptr;
    Sculpt* sculpt_to_test = nullptr;
    uint32_t model_to_test_idx = 0u;
    struct sGPU_RayData {
        glm::vec3   ray_origin;
        uint32_t    padd0;
        glm::vec3   ray_direction;
        uint32_t    padd1;
    } ray_to_upload;

    Uniform         ray_info_uniform;
    Uniform         ray_sculpt_instances_uniform;
    WGPUBindGroup   ray_sculpt_info_bind_group = nullptr;

    Uniform         ray_intersection_info_uniform;
    WGPUBindGroup   ray_intersection_info_bind_group = nullptr;

    Pipeline        ray_intersection_pipeline;
    Shader*         ray_intersection_shader = nullptr;

    Pipeline        ray_intersection_result_and_clean_pipeline;
    Shader*         ray_intersection_result_and_clean_shader = nullptr;

    Uniform         evaluation_job_result_count_uniform;
    Uniform         evaluation_aabb_culling_count_uniform;
    Uniform         evaluation_culling_dispatch_uniform;
    //Uniform         evaluation_write_to_tex_count_uniform;
    Uniform         evaluation_write_to_tex_buffer_uniform;

    Uniform         evaluator_num_bricks_by_wg_uniform;
    Uniform         evaluation_culling_dispatch_alt_uniform;
    Uniform         evaluation_write_to_tex_buffer_alt_uniform;

    Uniform         aabb_calculation_temp_buffer;
    WGPUBindGroup   aabb_calculation_temp_bind_group = nullptr;

    // Sculpt evaluation
    Pipeline write_to_texture_pipeline;
    Pipeline brick_removal_pipeline;
    Pipeline brick_copy_aabb_gen_pipeline;
    Pipeline brick_unmark_pipeline;
    Pipeline evaluator_1_aabb_culling_step_pipeline;
    Pipeline evaluator_1_5_interval_culling_step_pipeline;
    Pipeline evaluator_2_interval_culling_step_pipeline;
    Pipeline evaluator_2_5_write_to_texture_setup_pipeline;

    Shader* write_to_texture_shader = nullptr;
    Shader* brick_removal_shader = nullptr;
    Shader* brick_copy_aabb_gen_shader = nullptr;
    Shader* brick_unmark_shader = nullptr;
    Shader* evaluator_1_aabb_culling_step_shader = nullptr;
    Shader* evaluator_1_5_interval_culling_step_shader = nullptr;
    Shader* evaluator_2_interval_culling_step_shader = nullptr;
    Shader* evaluator_2_5_write_to_texture_setup_step_shader = nullptr;

    WGPUBindGroup   sdf_atlases_sampler_bindgroup = nullptr;
    WGPUBindGroup   indirect_brick_removal_bind_group = nullptr;
    WGPUBindGroup   sculpt_delete_bindgroup = nullptr;
    WGPUBindGroup   brick_unmark_bind_group = nullptr;
    WGPUBindGroup   preview_stroke_bind_group = nullptr;
    WGPUBindGroup   evaluator_aabb_culling_step_bind_group = nullptr;
    WGPUBindGroup   evaluator_stroke_history_bind_group = nullptr;
    WGPUBindGroup   evaluator_interval_culling_step_bind_group = nullptr;
    WGPUBindGroup   evaluator_write_to_texture_setup_bind_group = nullptr;
    WGPUBindGroup   evaluator_write_to_texture_step_bind_group = nullptr;

    // Evaluator uniforms
    uint32_t        octree_edit_list_size = 0u;
    Uniform         octree_edit_list;

    uint32_t        stroke_context_list_size = 0u;
    Uniform         stroke_context_list;

    Uniform         stroke_culling_buffer;

    Uniform         octant_usage_ping_pong_uniforms[4];
    Uniform         octant_usage_initialization_uniform[2];
    WGPUBindGroup   octant_usage_ping_pong_bind_groups[2] = {};

    bool reading_gpu_results = false;
    uint32_t frame_of_last_gpu_read = 0;
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
    void upload_strokes_and_edits(const uint32_t stroke_count, const std::vector<sGPUStroke>& strokes_to_compute, const uint32_t edits_count, const std::vector<Edit>& edits_to_upload);

    void init_shaders();
    void init_uniforms();
    void init_pipelines_and_bindgroups();

    bool evaluate(WGPUComputePassEncoder compute_pass, const sEvaluateRequest& evaluate_request);
    void evaluate_preview(WGPUComputePassEncoder compute_pass);

    void evaluate_closest_ray_intersection(WGPUComputePassEncoder compute_pass);

    void delete_sculpt(WGPUComputePassEncoder compute_pass, Sculpt* to_delete);

public:

    sGPU_SculptResults loaded_results;

    void init();
    void clean();

    void read_GPU_results();

    // TODO cleaning and deleting in frames
    void update(WGPUCommandEncoder command_encoder);

    void update_sculpt(Sculpt* sculpt, const sStrokeInfluence& strokes_to_process, const uint32_t edit_count, const std::vector<Edit>& edits_to_process);
    void set_preview_stroke(Sculpt* sculpt, const uint32_t in_gpu_model_idx, sGPUStroke preview_stroke, const std::vector<Edit>& preview_edits);

    void set_ray_to_test(const glm::vec3& ray_origin, const glm::vec3& ray_dir, SculptNode* node_to_test = nullptr);
    void set_ray_to_test(const glm::vec3& ray_origin, const glm::vec3& ray_dir, Sculpt * sculpt, const uint32_t model_id);


    Sculpt* create_sculpt();
    Sculpt* create_sculpt_from_history(const std::vector<Stroke>& stroke_history);

    bool has_performed_evaluation() { return performed_evaluation; }

    void delete_sculpt(Sculpt* to_delete) {
        sculpts_to_delete.push_back(to_delete);
    }
};
