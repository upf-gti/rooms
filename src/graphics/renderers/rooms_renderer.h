#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"

#include "raymarching_renderer.h"

// #define DISABLE_RAYMARCHER

class SculptManager;

struct sSDFGlobals {
    Uniform         brick_buffers;
    Uniform         brick_copy_buffer;

    Uniform         indirect_buffers;

    Texture         sdf_texture;
    Uniform         sdf_texture_uniform;

    Texture         sdf_material_texture;
    Uniform         sdf_material_texture_uniform;

    Uniform         preview_stroke_uniform_2;
    Uniform         preview_stroke_uniform;

    Uniform         linear_sampler_uniform;

    // Octree creation params
    uint8_t         octree_depth = 0;
    uint32_t        octants_max_size = 0;
    uint32_t        octree_total_size = 0;
    uint32_t        octree_last_level_size = 0u;
    uint32_t        max_brick_count = 0u;
    uint32_t        empty_brick_and_removal_buffer_count = 0u;
    float           brick_world_size = 0.0f;

    void clean();
};

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;
    SculptManager*      sculpt_manager = nullptr;

    float last_evaluation_time = 0.0f;

    sSDFGlobals sdf_globals;

    struct ProxyInstanceData {
        glm::vec3 position;
        uint32_t atlas_index;
        uint32_t octree_parent_index;
        uint32_t padding[3];
    };

    struct sSculptInstanceData {
        uint32_t flags = 0u;
        uint32_t instance_id;
        uint32_t pad0;
        uint32_t pad1;
        glm::mat4x4 model;
        glm::mat4x4 inv_model;
    };

    struct sSculptRenderInstances {
        Sculpt* sculpt = nullptr;
        uint16_t instance_count = 0u;
        sSculptInstanceData models[MAX_INSTANCES_PER_SCULPT];
    };

    std::map<uint32_t, sSculptRenderInstances*> sculpts_render_lists;
    std::vector<sSculptInstanceData> models_for_upload;
    Uniform         global_sculpts_instance_data_uniform;

    struct {
        uint32_t                count = 20u;
        std::vector<uint32_t>   count_buffer;
        Uniform                 uniform_count_buffer;
        WGPUBindGroup           count_bindgroup = nullptr;

        Pipeline                prepare_indirect;
        Shader* prepare_indirect_shader = nullptr;
    } sculpt_instances;


    void intialize_sculpt_render_instances();
    void upload_sculpt_models_and_instances(WGPUCommandEncoder command_encoder);
    void update_sculpts_indirect_buffers(WGPUCommandEncoder command_encoder);

public:

    RoomsRenderer();
    ~RoomsRenderer();

    virtual int pre_initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    virtual int initialize() override;
    virtual int post_initialize() override;
    void clean() override;

    void init_sdf_globals();

    void update(float delta_time) override;
    void render() override;

    uint32_t add_sculpt_render_call(Sculpt* sculpt, const glm::mat4& model, const uint32_t flags = 0u);

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }
    SculptManager* get_sculpt_manager() { return sculpt_manager; }

    inline Uniform& get_global_sculpts_instance_data() {
        return global_sculpts_instance_data_uniform;
    }

    inline std::map<uint32_t, sSculptRenderInstances*>& get_sculpts_render_list() {
        return sculpts_render_lists;
    }

    sSDFGlobals& get_sdf_globals() {
        return sdf_globals;
    }

    float get_last_evaluation_time() { return last_evaluation_time; }

    /*
    *   Edits
    */

    //void change_stroke(const StrokeParameters& params, const uint32_t index = 1u) {
    //    raymarching_renderer.change_stroke(params, index);
    //}

//    void push_edit_list(std::vector<Edit> &edits) {
//#ifndef DISABLE_RAYMARCHER
//        raymarching_renderer.push_edit_list(edits);
//#endif
//    };

    void push_preview_edit_list(std::vector<Edit>& edits) {
#ifndef DISABLE_RAYMARCHER
        for (uint32_t i = 0u; i < edits.size(); i++) {
            raymarching_renderer.add_preview_edit(edits[i]);
        }
#endif
    }

    inline void set_preview_edit(const Edit& stroke) {
        raymarching_renderer.add_preview_edit(stroke);
    }
};
