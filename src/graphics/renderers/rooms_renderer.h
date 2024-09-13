#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "raymarching_renderer.h"

// #define DISABLE_RAYMARCHER

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;

    float exposure = 1.0f;
    float ibl_intensity = 1.0f;

    void render_screen(WGPUTextureView swapchain_view);

    float last_evaluation_time = 0.0f;

#if defined(XR_SUPPORT)

    void render_xr();

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
