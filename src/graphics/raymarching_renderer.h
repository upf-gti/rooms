#pragma once

#include "includes.h"
#include "graphics/renderer.h"
#include <vector>
#include "edit.h"

#define EDITS_MAX 2048

class RaymarchingRenderer : public Renderer {

    // Render to screen
    WGPURenderPipeline      render_pipeline = nullptr;
    WGPUPipelineLayout      render_pipeline_layout = nullptr;
    Shader*                 render_shader = nullptr;

    WGPUBindGroup           render_bind_group_left_eye = nullptr;
    WGPUBindGroup           render_bind_group_right_eye = nullptr;

    Uniform                 u_render_texture_left_eye;
    Uniform                 u_render_texture_right_eye;

    // Compute
    WGPUComputePipeline     compute_raymarching_pipeline = nullptr;
    Shader*                 compute_raymarching_shader = nullptr;
    WGPUBindGroup           compute_raymarching_textures_bind_group = nullptr;
    WGPUBindGroup           compute_raymarching_data_bind_group = nullptr;

    WGPUComputePipeline     compute_merge_pipeline = nullptr;
    Shader*                 compute_merge_shader = nullptr;
    WGPUBindGroup           compute_merge_bind_group = nullptr;

    WGPUTexture             left_eye_texture = nullptr;
    WGPUTexture             right_eye_texture = nullptr;

    //WGPUTexture             sdf_texture = nullptr;

    Uniform                 u_compute_buffer_data;
    Uniform                 u_compute_texture_left_eye;
    Uniform                 u_compute_texture_right_eye;

    Uniform                 u_compute_texture_sdf_storage;
    //Uniform                 u_compute_texture_sdf_read;

    Uniform                 u_compute_edits_data;

    // Data needed for XR raymarching
    struct sComputeData {
        glm::mat4x4 inv_view_projection_left_eye;
        glm::mat4x4 inv_view_projection_right_eye;

        glm::vec3 left_eye_pos;
        float render_height = 0.0f;
        glm::vec3 right_eye_pos;
        float render_width = 0.0f;

        float time = 0.0f;
        float dummy0 = 0.0f;
        float dummy1 = 0.0f;
        float dummy2 = 0.0f;
    };

    sComputeData                      compute_data;
    Edit                              edits[EDITS_MAX];

    Mesh                              quad_mesh;
    WGPUVertexBufferLayout            quad_vertex_layout;
    std::vector<WGPUVertexAttribute>  quad_vertex_attributes;
    WGPUBuffer                        quad_vertex_buffer = nullptr;

    // For the XR mirror screen
#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    WGPURenderPipeline      mirror_pipeline;
    Shader* mirror_shader;
#endif

    void render(WGPUTextureView swapchain_view, WGPUBindGroup bind_group);
    void render_screen();

#if defined(XR_SUPPORT)
    void render_xr();
#endif

    void compute();

    void init_render_pipeline();
    void init_compute_raymarching_pipeline();
    void init_compute_merge_pipeline();

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    void render_mirror();
    void init_mirror_pipeline();
#endif

public:

    RaymarchingRenderer();

    virtual int initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    virtual void clean() override;

    virtual void update(float delta_time) override;
    virtual void render() override;

};