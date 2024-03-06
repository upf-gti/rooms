#pragma once

#include "includes.h"

#include "graphics/pipeline.h"

class MeshRenderer {

    // Render meshes with material color
    WGPUBindGroup           render_bind_group_camera = nullptr;
    Shader*                 render_mesh_shader = nullptr;

    void init_render_mesh_pipelines();

public:

    MeshRenderer();

    int initialize();
    void clean();

    void update(float delta_time);
    void render_opaque(WGPURenderPassEncoder render_pass);
    void render_transparent(WGPURenderPassEncoder render_pass);

};
