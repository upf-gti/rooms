#pragma once

#include "framework/resources/sculpt.h"
#include "graphics/texture.h"

class SculptManager {
    uint32_t        sculpt_count = 0u;

    std::vector<Sculpt*> sculpts_to_delete;
    std::vector<Sculpt*> sculpts_to_clean;
    std::vector<Sculpt*> sculpts_to_create;

    bool needs_evaluation = false;

    // Sculpt deletion
    Pipeline        sculpt_delete_pipeline;
    Shader* sculpt_delete_shader = nullptr;
    WGPUBindGroup   compute_octree_clean_octree_bind_group = nullptr;

    // Evaluation initialization
    Pipeline        evaluation_initialization_pipeline;
    Shader* evaluation_initialization_shader = nullptr;
    WGPUBindGroup   evaluation_initialization_bind_group = nullptr;

    // Sculpt evaluation
    Pipeline evaluate_pipeline;
    Pipeline increment_level_pipeline;
    Pipeline write_to_texture_pipeline;
    Pipeline brick_removal_pipeline;
    Pipeline brick_copy_pipeline;

    Shader* evaluate_shader = nullptr;
    Shader* increment_level_shader = nullptr;
    Shader* write_to_texture_shader = nullptr;
    Shader* brick_removal_shader = nullptr;
    Shader* brick_copy_shader = nullptr;

    WGPUBindGroup   evaluate_bind_group = nullptr;
    WGPUBindGroup   increment_level_bind_group = nullptr;
    WGPUBindGroup   write_to_texture_bind_group = nullptr;
    WGPUBindGroup   indirect_brick_removal_bind_group = nullptr;
    WGPUBindGroup   brick_copy_bind_group = nullptr;
    WGPUBindGroup   sculpt_delete_bindgroup = nullptr;

    // Evaluator uniforms
    Uniform         merge_data_uniform;

    uint32_t        octree_edit_list_size = 0u;
    Uniform         octree_edit_list;

    uint32_t        stroke_context_list_size = 0u;
    Uniform         stroke_context_list;

    Uniform         stroke_culling_buffer;

    Uniform         octant_usage_ping_pong_uniforms[4];
    Uniform         octant_usage_initialization_uniform[2];
    WGPUBindGroup   octant_usage_ping_pong_bind_groups[2] = {};

    Uniform         octree_indirect_buffer_struct;

    // Octree struct reference
    struct sOctreeNode {
        glm::vec2 octant_center_distance = glm::vec2(10000.0f, 10000.0f);
        uint32_t dummy = 0.0f;
        uint32_t tile_pointer = 0;
        glm::vec3 padding;
        uint32_t culling_data = 0u;
    };

    // Data needed for sdf merging
    struct sMergeData {
        glm::vec3  reevaluation_AABB_min;
        uint32_t   reevaluate = 0u;
        glm::vec3  reevaluation_AABB_max;
        uint32_t   padding;
    } compute_merge_data;

    void upload_strokes_and_edits(const std::vector<sToUploadStroke>& strokes_to_compute, const std::vector<Edit>& edits_to_upload);

    void init_uniforms();
    void load_shaders();
    void init_pipelines_and_bindgroups();
    //struct sTodo{
    //    Sculpt
    //    AABB
    //    list<Stroke>
    //};
public:
    void init();
    void clean();

    // TODO cleaning and delteing in frames
    void update();

    void compute_octree(WGPUCommandEncoder command_encoder, bool show_preview);

    Sculpt* create_sculpt();
    Sculpt* create_sculpt_from_history(const std::vector<Stroke>& stroke_history);

    void delete_sculpt(WGPUComputePassEncoder compute_pass, Sculpt* to_delete);

    void evaluate(WGPUComputePassEncoder compute_pass, Sculpt* sculpt, const std::vector<sToUploadStroke>& stroke_to_eval, const std::vector<Edit>& edits_to_upload, const AABB& stroke_aabb);
};
