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

    // Octree creation params
    uint8_t         octree_depth = 0;
    uint32_t        octants_max_size = 0;
    uint32_t        octree_total_size = 0;
    uint32_t        octree_last_level_size = 0u;
    uint32_t        max_brick_count = 0u;
    uint32_t        empty_brick_and_removal_buffer_count = 0u;
    float           brick_world_size = 0.0f;
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

public:

    RoomsRenderer();
    ~RoomsRenderer();

    int initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    void clean() override;

    void init_sdf_globals();

    void update(float delta_time) override;
    void render() override;

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }
    SculptManager* get_sculpt_manager() { return sculpt_manager; }

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
