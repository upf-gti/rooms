#pragma once

#include "includes.h"
#include "graphics/renderer.h"
#include <vector>
#include "graphics/edit.h"

#include "graphics/texture.h"

#include "tools/sculpt/tool.h"

#include "raymarching_renderer.h"
#include "mesh_renderer.h"

#ifdef __EMSCRIPTEN__
#define DISABLE_RAYMARCHER
#endif

#define EDITS_MAX 1024
#define SDF_RESOLUTION 512

enum eEYE {
    EYE_LEFT,
    EYE_RIGHT,
    EYE_SIZE // Let's assume this will never be different to 2...
};

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;
    MeshRenderer mesh_renderer;

    glm::vec3 clear_color;
    Mesh  quad_mesh;

    // Render to screen
    Pipeline render_quad_pipeline;
    Shader*  render_quad_shader = nullptr;

    WGPUBindGroup   eye_render_bind_group[EYE_SIZE] = {};
    WGPUTextureView eye_depth_texture_view[EYE_SIZE] = {};

    void render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group);
    void render_screen();

    void init_render_quad_pipeline();
    void init_render_quad_bind_groups();

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

    RoomsRenderer();

    virtual int initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    virtual void clean() override;

    virtual void update(float delta_time) override;
    virtual void render() override;

    void resize_window(int width, int height) override;

    void set_sculpt_start_position(const glm::vec3& position) {
        raymarching_renderer.set_sculpt_start_position(position);
    }

    /*
    *   Edits
    */

    void push_edit(Edit edit) {
        raymarching_renderer.push_edit(edit);
    };

    void push_edit_list(std::vector<Edit> &edits) {
        raymarching_renderer.push_edit_list(edits);
    };

    void set_preview_edit(const Edit& edit) {
        raymarching_renderer.set_preview_edit(edit);
    }

    void set_sculpt_rotation(const glm::quat& rotation) {
        raymarching_renderer.set_sculpt_rotation(rotation);
    }

};
