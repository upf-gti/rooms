#pragma once

#include "includes.h"

#include "graphics/pipeline.h"

#include "tools/sculpt/tool.h"

class MeshRenderer {

    // Render meshes with material color
    Pipeline                render_mesh_pipeline;
    WGPUBindGroup           render_bind_group_camera = nullptr;
    Shader*                 render_mesh_shader = nullptr;

    // Render meshes with textures
    Pipeline                render_mesh_texture_pipeline;
    Shader*                 render_mesh_texture_shader = nullptr;

    // Render UI Quads
    Pipeline                render_mesh_ui_pipeline;
    Shader*                 render_mesh_ui_shader = nullptr;
    Pipeline                render_mesh_ui_texture_pipeline;
    Shader*                 render_mesh_ui_texture_shader = nullptr;

    // Render Mesh Grid
    Pipeline                render_mesh_grid_pipeline;
    Shader*                 render_mesh_grid_shader = nullptr;

    // Font rendering
    Pipeline                render_fonts_pipeline;
    Shader*                 render_fonts_shader = nullptr;

    Uniform                 camera_uniform;

    // For the XR mirror screen
#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    Pipeline mirror_pipeline;
    Shader*  mirror_shader = nullptr;

    std::vector<Uniform> swapchain_uniforms;
    std::vector<WGPUBindGroup> swapchain_bind_groups;
#endif

    void init_render_mesh_pipelines();

public:

    MeshRenderer();

    int initialize();
    void clean();

    void clean_renderables();

    void set_view_projection(const glm::mat4x4& view_projection);

    void update(float delta_time);
    void render(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth);

};
