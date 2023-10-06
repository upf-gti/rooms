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

void MeshRenderer::set_view_projection(const glm::mat4x4& view_projection)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &(view_projection), sizeof(view_projection));
}

void MeshRenderer::update(float delta_time)
{

}

void MeshRenderer::render(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Prepare the color attachment
    WGPURenderPassColorAttachment render_pass_color_attachment = {};
    render_pass_color_attachment.view = swapchain_view;
#ifndef DISABLE_RAYMARCHER
    render_pass_color_attachment.loadOp = WGPULoadOp_Load;
#else
    render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
#endif
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;

    glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
    render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

    // Prepate the depth attachment
    WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
    render_pass_depth_attachment.view = swapchain_depth;
    render_pass_depth_attachment.depthClearValue = 1.0f;
#ifndef DISABLE_RAYMARCHER
    render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Load;
#else
    render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Clear;
#endif
    render_pass_depth_attachment.depthStoreOp = WGPUStoreOp_Store;
    render_pass_depth_attachment.depthReadOnly = false;
    render_pass_depth_attachment.stencilClearValue = 0; // Stencil config necesary, even if unused
    render_pass_depth_attachment.stencilLoadOp = WGPULoadOp_Undefined;
    render_pass_depth_attachment.stencilStoreOp = WGPUStoreOp_Undefined;
    render_pass_depth_attachment.stencilReadOnly = true;

    WGPURenderPassDescriptor render_pass_descr = {};
    render_pass_descr.colorAttachmentCount = 1;
    render_pass_descr.colorAttachments = &render_pass_color_attachment;
    render_pass_descr.depthStencilAttachment = &render_pass_depth_attachment;

    // Create & fill the render pass (encoder)
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

    RoomsRenderer::instance->render(render_pass, render_bind_group_camera);

    wgpuRenderPassEncoderEnd(render_pass);

    wgpuRenderPassEncoderRelease(render_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(command_encoder);
}

void MeshRenderer::init_render_mesh_pipelines()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    render_mesh_shader = RendererStorage::get_shader("data/shaders/mesh_color.wgsl");
    Shader* render_mesh_texture_shader = RendererStorage::get_shader("data/shaders/mesh_texture.wgsl");
    Shader* render_mesh_ui_shader = RendererStorage::get_shader("data/shaders/mesh_ui.wgsl");
    Shader* render_mesh_ui_texture_shader = RendererStorage::get_shader("data/shaders/mesh_texture_ui.wgsl");
    Shader* render_fonts_shader = RendererStorage::get_shader("data/shaders/sdf_fonts.wgsl");
    Shader* render_mesh_grid_shader = RendererStorage::get_shader("data/shaders/mesh_grid.wgsl");

    // Camera

    camera_uniform.data = webgpu_context->create_buffer(sizeof(glm::mat4x4), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    camera_uniform.binding = 0;
    camera_uniform.buffer_size = sizeof(glm::mat4x4);

    std::vector<Uniform*> uniforms = { &camera_uniform };

    render_bind_group_camera = webgpu_context->create_bind_group(uniforms, render_mesh_shader, 1);

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUBlendState blend_state;
    blend_state.color = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    };
    blend_state.alpha = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One,
    };

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = &blend_state;
    color_target.writeMask = WGPUColorWriteMask_All;

    Pipeline::register_render_pipeline(render_mesh_shader, color_target, true);
    Pipeline::register_render_pipeline(render_mesh_texture_shader, color_target, true);
    Pipeline::register_render_pipeline(render_mesh_ui_shader, color_target, true);
    Pipeline::register_render_pipeline(render_mesh_ui_texture_shader, color_target, true);
    Pipeline::register_render_pipeline(render_fonts_shader, color_target, true);
    Pipeline::register_render_pipeline(render_mesh_grid_shader, color_target, true);
}
