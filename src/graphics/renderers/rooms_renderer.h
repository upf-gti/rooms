#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "raymarching_renderer.h"
#include "mesh_renderer.h"

#include "framework/camera/flyover_camera.h"

#ifdef __EMSCRIPTEN__
#define DISABLE_RAYMARCHER
#endif

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;

    FlyoverCamera camera;

    Mesh  quad_mesh;
    Uniform         camera_uniform;

    // Render to screen
    Pipeline        render_quad_pipeline = {};
    Shader*         render_quad_shader = nullptr;

    Texture         eye_textures[EYE_COUNT] = {};
    Texture         eye_depth_textures[EYE_COUNT] = {};

    Uniform         eye_render_texture_uniform[EYE_COUNT] = {};

    WGPUBindGroup   eye_render_bind_group[EYE_COUNT] = {};
    WGPUTextureView eye_depth_texture_view[EYE_COUNT] = {};

    void render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group);
    void render_screen();

    void init_render_quad_pipeline();
    void init_render_quad_bind_groups();
    void init_camera_bindgroup();
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

    inline Uniform* get_current_camera_uniform() {
        return &camera_uniform;
    }

    void set_sculpt_start_position(const glm::vec3& position) {
        raymarching_renderer.set_sculpt_start_position(position);
    }

    Texture* get_eye_texture(eEYE eye);

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }

    /*
    *   Edits
    */

    void push_edit(Edit edit) {
        raymarching_renderer.push_edit(edit);
    };

    void push_edit_list(std::vector<Edit> &edits) {
        raymarching_renderer.push_edit_list(edits);
    };

    void add_preview_edit(const Edit& edit) {
        raymarching_renderer.add_preview_edit(edit);
    }

    void push_preview_edit_list(std::vector<Edit>& edits) {
        for (uint32_t i = 0u; i < edits.size(); i++) {
            raymarching_renderer.add_preview_edit(edits[i]);
        }
    }

    void set_sculpt_rotation(const glm::quat& rotation) {
        raymarching_renderer.set_sculpt_rotation(rotation);
    }

};
