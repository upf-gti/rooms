#pragma once

#include "includes.h"

#include "graphics/pipeline.h"

#include "tools/sculpt/tool.h"

class MeshRenderer {

    // Render meshes with material color
    WGPUBindGroup           render_bind_group_camera = nullptr;
    Shader*                 render_mesh_shader = nullptr;

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

    void update(float delta_time);
    void render_opaque(WGPURenderPassEncoder render_pass);
    void render_transparent(WGPURenderPassEncoder render_pass);

};
