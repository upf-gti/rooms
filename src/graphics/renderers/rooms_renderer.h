#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "raymarching_renderer.h"
#include "mesh_renderer.h"

#include "framework/camera/flyover_camera.h"
#include "framework/camera/orbit_camera.h"

#ifdef __EMSCRIPTEN__
#define DISABLE_RAYMARCHER
#endif

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;

    Mesh                quad_mesh;
    Uniform             camera_uniform;

    struct sCameraData {
        glm::mat4x4 mvp;
        glm::vec3 eye;
        float dummy;
    } camera_data;

    Texture         eye_depth_textures[EYE_COUNT] = {};
    WGPUTextureView eye_depth_texture_view[EYE_COUNT] = {};

    void render_screen();

    void init_depth_buffers();
    void init_camera_bind_group();

#if defined(XR_SUPPORT)

    void render_xr();

    // For the XR mirror screen
#if defined(USE_MIRROR_WINDOW)
    void render_mirror();
    void init_mirror_pipeline();

    Pipeline mirror_pipeline;
    Shader* mirror_shader = nullptr;

    std::vector<Uniform> swapchain_uniforms;
    std::vector<WGPUBindGroup> swapchain_bind_groups;
#endif // USE_MIRROR_WINDOW

#endif // XR_SUPPORT

public:
    MeshRenderer mesh_renderer;


    RoomsRenderer();

    int initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    void resize_window(int width, int height) override;

    inline Uniform* get_current_camera_uniform() { return &camera_uniform; }

    void set_sculpt_start_position(const glm::vec3& position) {
        raymarching_renderer.set_sculpt_start_position(position);
    }

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }

    /*
    *   Edits
    */

    void change_stroke(const StrokeParameters& params) {
        raymarching_renderer.change_stroke(params);
    }

    void push_edit(Edit edit) {
        raymarching_renderer.push_edit(edit);
    };

    void push_edit_list(std::vector<Edit> &edits) {
#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.push_edit_list(edits);
#endif
    };

    void add_preview_edit(const Edit& edit) {
#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.add_preview_edit(edit);
#endif
    }

    void push_preview_edit_list(std::vector<Edit>& edits) {
#ifndef DISABLE_RAYMARCHER
        for (uint32_t i = 0u; i < edits.size(); i++) {
            raymarching_renderer.add_preview_edit(edits[i]);
        }
#endif
    }

    void set_sculpt_rotation(const glm::quat& rotation) {
        raymarching_renderer.set_sculpt_rotation(rotation);
    }

    void undo() {
        raymarching_renderer.undo();
    }

    void redo() {
        raymarching_renderer.redo();
    }

    glm::vec3 get_camera_eye();
};
