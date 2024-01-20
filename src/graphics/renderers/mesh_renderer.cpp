#include "mesh_renderer.h"

#include "rooms_renderer.h"

MeshRenderer::MeshRenderer()
{
    
}

int MeshRenderer::initialize()
{
    init_render_mesh_pipelines();

    return 0;
}

void MeshRenderer::clean()
{

}

void MeshRenderer::update(float delta_time)
{

}

void MeshRenderer::render_opaque(WGPURenderPassEncoder render_pass)
{
    RoomsRenderer::instance->render_opaque(render_pass, render_bind_group_camera);
}

void MeshRenderer::render_transparent(WGPURenderPassEncoder render_pass)
{
    RoomsRenderer::instance->render_transparent(render_pass, render_bind_group_camera);
}

void MeshRenderer::init_render_mesh_pipelines()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    render_mesh_shader = RendererStorage::get_shader("data/shaders/mesh_color.wgsl");

    // Camera
    std::vector<Uniform*> uniforms = { dynamic_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_current_camera_uniform() };

    render_bind_group_camera = webgpu_context->create_bind_group(uniforms, render_mesh_shader, 1);

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUBlendState* blend_state = new WGPUBlendState;
    blend_state->color = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    };
    blend_state->alpha = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One,
    };

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = blend_state;
    color_target.writeMask = WGPUColorWriteMask_All;

    Pipeline::register_render_pipeline(render_mesh_shader, color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/mesh_texture.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/mesh_texture_cube.wgsl"), color_target, { .cull_mode = WGPUCullMode_Front, .uses_depth_write = false });
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/mesh_grid.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/mesh_transparent.wgsl"), color_target, { .cull_mode = WGPUCullMode_Front });
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/mesh_outline.wgsl"), color_target, { .cull_mode = WGPUCullMode_Front });
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_group.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_button.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_button.wgsl", { "USES_TEXTURE" }), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_slider.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_slider_h.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/ui/ui_color_picker.wgsl"), color_target);
    Pipeline::register_render_pipeline(RendererStorage::get_shader("data/shaders/sdf_fonts.wgsl"), color_target);
}
