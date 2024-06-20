#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"
#include "graphics/texture.h"
#include "graphics/surface.h"

#include "raymarching_renderer.h"

#include "framework/camera/flyover_camera.h"
#include "framework/camera/orbit_camera.h"

// #define DISABLE_RAYMARCHER

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;

    Surface quad_surface;

    Uniform camera_uniform;
    Uniform camera_2d_uniform;
    Uniform linear_sampler_uniform;

    uint32_t camera_buffer_stride = 0;

    WGPUCommandEncoder global_command_encoder;

    struct sCameraData {
        glm::mat4x4 mvp;

        glm::vec3 eye;
        float exposure;

        glm::vec3 right_controller_position;
        float ibl_intensity;
    };

    sCameraData camera_data;
    sCameraData camera_2d_data;

    float exposure = 1.0f;
    float ibl_intensity = 1.0f;

    void render_screen(WGPUTextureView swapchain_view);

    // Render meshes with material color
    WGPUBindGroup render_bind_group_camera = nullptr;
    WGPUBindGroup render_bind_group_camera_2d = nullptr;

    float last_evaluation_time = 0.0f;

    void init_camera_bind_group();

#if defined(XR_SUPPORT)

    void render_xr();

    // For the XR mirror screen
#if defined(USE_MIRROR_WINDOW)
    void render_mirror(WGPUTextureView swapchain_view);
    void init_mirror_pipeline();

    Pipeline mirror_pipeline;
    Shader* mirror_shader = nullptr;

    std::vector<Uniform> swapchain_uniforms;
    std::vector<WGPUBindGroup> swapchain_bind_groups;
#endif // USE_MIRROR_WINDOW

#endif // XR_SUPPORT

    bool debug_this_frame = false;

public:

    RoomsRenderer();
    ~RoomsRenderer();

    int initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    void resize_window(int width, int height) override;

    inline void set_exposure(float new_exposure) { exposure = new_exposure; }
    inline void set_ibl_intensity(float new_intensity) { ibl_intensity = new_intensity; }

    inline void toogle_frame_debug() { debug_this_frame = true; }

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }
    inline Uniform* get_current_camera_uniform() { return &camera_uniform; }
    glm::vec3 get_camera_eye();
    glm::vec3 get_camera_front();
    float get_last_evaluation_time() { return last_evaluation_time; }
    float get_exposure() { return exposure; }
    float get_ibl_intensity() { return ibl_intensity; }

    /*
    *   Edits
    */

    void change_stroke(const StrokeParameters& params, const uint32_t index = 1u) {
        raymarching_renderer.change_stroke(params, index);

    }

    void push_edit(Edit edit) {
        raymarching_renderer.push_edit(edit);
    };

    void push_edit_list(std::vector<Edit> &edits) {
#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.push_edit_list(edits);
#endif
    };

    void push_preview_edit_list(std::vector<Edit>& edits) {
#ifndef DISABLE_RAYMARCHER
        for (uint32_t i = 0u; i < edits.size(); i++) {
            raymarching_renderer.add_preview_edit(edits[i]);
        }
#endif
    }

    void undo() {
        raymarching_renderer.undo();
    }

    void redo() {
        raymarching_renderer.redo();
    }

    inline void set_preview_edit(const Edit& stroke) {
        raymarching_renderer.add_preview_edit(stroke);
    }
};
