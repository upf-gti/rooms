#pragma once

#include "graphics/pipeline.h"
#include "framework/resources/resource.h"
#include "graphics/edit.h"


#include <glm/glm.hpp>
#include <vector>

class Sculpt : public Resource {
public:
    Sculpt(const uint32_t id, const Uniform uniform, const WGPUBindGroup bindgroup):
        sculpt_id (id), octree_uniform(uniform), octree_bindgroup(bindgroup) {};
    uint32_t sculpt_id;

    Uniform octree_uniform;
    WGPUBindGroup octree_bindgroup = nullptr;

    std::vector<Stroke> stroke_history;

    void init();
    void clean();
};

class SculptManager {
    uint32_t        sculpt_count = 0u;

    // Stroke culling data
    uint32_t        max_stroke_influence_count = 100u;

    // Sculpt deletion
    Pipeline        sculpt_delete_pipeline;
    Shader*         sculpt_delete_shader = nullptr;
    WGPUBindGroup   compute_octree_clean_octree_bind_group = nullptr;

    // Sculpt evaluation
    //TODO: Rename Evaluator intialization 
    Pipeline        evaluation_initialization_pipeline;
    Shader* evaluation_initialization_shader = nullptr;
    WGPUBindGroup   evaluation_initialization_bind_group = nullptr;

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

    uint32_t        stroke_context_size = 0u;
    Uniform         octree_stroke_context;

    Uniform         stroke_culling_data;


    // TODO: Same buffers with diferent bidings 
    Uniform         octant_usage_ping_pong_uniforms[4];
    Uniform         octant_usage_initialization_uniform[2];
    WGPUBindGroup   octant_usage_ping_pong_bind_groups[2] = {};


    Uniform         octree_indirect_buffer_struct;
    Uniform         octree_indirect_buffer_struct_2;

    struct {
        Uniform         brick_buffers;
        Uniform         brick_copy_buffer;

        Texture         sdf_texture;
        Uniform         sdf_texture_uniform;

        Texture         sdf_material_texture;
        Uniform         sdf_material_texture_uniform;

        // Octree creation params
        uint8_t         octree_depth = 0;
        uint32_t        octants_max_size = 0;
        uint32_t        octree_total_size = 0;
        uint32_t        octree_last_level_size = 0u;
        uint32_t        max_brick_count = 0u;
        uint32_t        empty_brick_and_removal_buffer_count = 0u;
        float           brick_world_size = 0.0f;
    } SDF_Globals;

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

public:
    void init();
    void clean();

    // TODO cleaning and delteing in frames
    void update();

    Sculpt* create_sculpt();
    void delete_sculpt(WGPUComputePassEncoder compute_pass, const Sculpt& to_delete);

    void evaluate(WGPUComputePassEncoder compute_pass, const Sculpt &sculpt, const std::vector<sToUploadStroke> &stroke_to_eval, const std::vector<Edit>& edits_to_upload, const AABB &stroke_aabb);
};
