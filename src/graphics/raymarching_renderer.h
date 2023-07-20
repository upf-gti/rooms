#pragma once

#include "includes.h"
#include "graphics/renderer.h"
#include <vector>
#include "edit.h"

#include "graphics/texture.h"

#define EDITS_MAX 1024
#define SDF_RESOLUTION 512

class RaymarchingRenderer : public Renderer {

    // Render to screen
    Pipeline                render_quad_pipeline;
    Shader*                 render_quad_shader = nullptr;

    WGPUBindGroup           render_bind_group_left_eye = nullptr;
    WGPUBindGroup           render_bind_group_right_eye = nullptr;

    Uniform                 u_render_texture_left_eye;
    Uniform                 u_render_texture_right_eye;

    // Compute
    Pipeline                initialize_sdf_pipeline;
    Shader*                 initialize_sdf_shader = nullptr;
    WGPUBindGroup           initialize_sdf_bind_group = nullptr;

    Pipeline                compute_raymarching_pipeline;
    Shader*                 compute_raymarching_shader = nullptr;
    WGPUBindGroup           compute_raymarching_textures_bind_group = nullptr;
    WGPUBindGroup           compute_raymarching_data_bind_group = nullptr;

    Pipeline                compute_merge_pipeline;
    Shader*                 compute_merge_shader = nullptr;
    WGPUBindGroup           compute_merge_bind_group = nullptr;

    Texture                 left_eye_texture;
    Texture                 left_eye_depth_texture;
    Texture                 right_eye_texture;
    Texture                 right_eye_depth_texture;

    WGPUTextureView         left_eye_depth_texture_view = nullptr;
    WGPUTextureView         right_eye_depth_texture_view = nullptr;

    // Render meshes with material color
    Pipeline                render_mesh_pipeline;
    WGPUBindGroup           render_bind_group_camera = nullptr;
    Shader*                 render_mesh_shader = nullptr;

    // Render meshes with textures
    Pipeline                render_mesh_texture_pipeline;
    Shader*                 render_mesh_texture_shader = nullptr;

    // Font rendering
    Pipeline                render_fonts_pipeline;
    Shader*                 render_fonts_shader = nullptr;

    Uniform                 u_camera;
    Uniform                 u_font;

    Uniform                 u_compute_buffer_data;
    Uniform                 u_compute_texture_left_eye;
    Uniform                 u_compute_texture_right_eye;

    Uniform                 u_compute_texture_sdf_storage;
    //Uniform                 u_compute_texture_sdf_read;

    Uniform                 u_compute_merge_data;
    Uniform                 u_compute_edits_array;

    // Data needed for XR raymarching
    struct sComputeData {
        glm::mat4x4 inv_view_projection_left_eye;
        glm::mat4x4 inv_view_projection_right_eye;

        glm::vec3 left_eye_pos;
        float render_height = 0.0f;
        glm::vec3 right_eye_pos;
        float render_width = 0.0f;

        float time = 0.0f;
        float camera_near = 0.0f;
        float camera_far = 0.0f;
        float dummy0 = 0.0f;
    } compute_raymarching_data;

    // Data needed for sdf merging
    struct sMergeData {
        glm::uvec3 sdf_size = {};
        uint32_t edits_to_process = 0;
    } compute_merge_data;

    struct sCameraData {
        glm::mat4x4 view_projection;
    } camera_data;

    sEdit edits[EDITS_MAX];

    Mesh  quad_mesh;

    // For the XR mirror screen
#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    Pipeline mirror_pipeline;
    Shader*  mirror_shader = nullptr;

    std::vector<Uniform> swapchain_uniforms;
    std::vector<WGPUBindGroup> swapchain_bind_groups;
#endif


    void render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group);
    void render_screen();

    void render_meshes(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth);

#if defined(XR_SUPPORT)
    void render_xr();
#endif

    void compute_initialize_sdf();
    void compute_merge();
    void compute_raymarching();

    void init_render_quad_pipeline();
    void init_render_mesh_pipelines();
    void init_compute_raymarching_pipeline();
    void init_initialize_sdf_pipeline();
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

    /*
    *   Edits
    */

    void push_edit(sEdit edit) {
        edits[compute_merge_data.edits_to_process++] = edit;
    };
};