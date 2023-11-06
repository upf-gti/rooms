#include "rooms_renderer.h"

#ifdef XR_SUPPORT
#include "dawnxr/dawnxr_internal.h"
#endif

RoomsRenderer::RoomsRenderer() : Renderer()
{
    
}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    init_render_quad_pipeline();
    init_camera_bindgroup();

    raymarching_renderer.initialize(use_mirror_screen);
    mesh_renderer.initialize();

#ifdef XR_SUPPORT
    if (is_openxr_available && use_mirror_screen) {
        init_mirror_pipeline();
    }
#endif

    camera.set_perspective(glm::radians(45.0f), webgpu_context.render_width / static_cast<float>(webgpu_context.render_height), z_near, z_far);
    camera.look_at(glm::vec3(0.0f, 0.6f, 0.6f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera.set_mouse_sensitivity(0.01f);

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    raymarching_renderer.clean();
    mesh_renderer.clean();

    eye_render_texture_uniform[EYE_LEFT].destroy();
    eye_render_texture_uniform[EYE_RIGHT].destroy();

    wgpuBindGroupRelease(eye_render_bind_group[EYE_LEFT]);
    wgpuBindGroupRelease(eye_render_bind_group[EYE_RIGHT]);

    wgpuTextureViewRelease(eye_depth_texture_view[EYE_LEFT]);
    wgpuTextureViewRelease(eye_depth_texture_view[EYE_RIGHT]);

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    if (is_openxr_available) {
        for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
            swapchain_uniforms[i].destroy();
            wgpuBindGroupRelease(swapchain_bind_groups[i]);
        }
    }
#endif
}

void RoomsRenderer::update(float delta_time)
{
    camera.update(delta_time);

    raymarching_renderer.update(delta_time);
    mesh_renderer.update(delta_time);
}

void RoomsRenderer::render()
{
    if (!is_openxr_available) {
        render_screen();
    }

#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        render_xr();

        if (use_mirror_screen) {
            render_mirror();
        }
    }
#endif

    clear_renderables();
}

void RoomsRenderer::render_screen()
{
    const glm::mat4x4& view_projection = camera.get_view_projection();

    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &(view_projection), sizeof(view_projection));

    //raymarching_renderer.compute_raymarching();

    WGPUTextureView swapchain_view = wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain);

    //render_eye_quad(swapchain_view, eye_depth_texture_view[EYE_LEFT], eye_render_bind_group[EYE_LEFT]);

#ifndef DISABLE_RAYMARCHER
    raymarching_renderer.set_camera_eye(camera.get_eye());
    raymarching_renderer.render_raymarching_proxy(swapchain_view, eye_depth_texture_view[EYE_LEFT]);
#endif

    mesh_renderer.render(swapchain_view, eye_depth_texture_view[EYE_LEFT]);
    
    wgpuTextureViewRelease(swapchain_view);

#ifndef __EMSCRIPTEN__
    wgpuSwapChainPresent(webgpu_context.screen_swapchain);
#endif
}

#if defined(XR_SUPPORT)

void RoomsRenderer::render_xr()
{
    prepare_instancing();

    xr_context.init_frame();

    //raymarching_renderer.set_left_eye(xr_context.per_view_data[EYE_LEFT].position, xr_context.per_view_data[EYE_LEFT].view_projection_matrix);
    //raymarching_renderer.set_right_eye(xr_context.per_view_data[EYE_RIGHT].position, xr_context.per_view_data[EYE_RIGHT].view_projection_matrix);
    //raymarching_renderer.set_near_far(xr_context.z_near, xr_context.z_far);

    //raymarching_renderer.compute_raymarching();

    for (uint32_t i = 0; i < xr_context.view_count; ++i) {

        xr_context.acquire_swapchain(i);

        const sSwapchainData& swapchainData = xr_context.swapchains[i];

        render_eye_quad(swapchainData.images[swapchainData.image_index].textureView, eye_depth_texture_view[i], eye_render_bind_group[i]);

        WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &(xr_context.per_view_data[i].view_projection_matrix), sizeof(xr_context.per_view_data[i].view_projection_matrix));

        raymarching_renderer.set_camera_eye(xr_context.per_view_data[i].position);
        raymarching_renderer.render_raymarching_proxy(swapchainData.images[swapchainData.image_index].textureView, eye_depth_texture_view[i]);

        mesh_renderer.render(swapchainData.images[swapchainData.image_index].textureView, eye_depth_texture_view[i]);

        xr_context.release_swapchain(i);
    }

    xr_context.end_frame();
}
#endif

void RoomsRenderer::render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group)
{
    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Prepare the color attachment
    WGPURenderPassColorAttachment render_pass_color_attachment = {};
    render_pass_color_attachment.view = swapchain_view;
    render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
    render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

    // Prepate the depth attachment
    WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
    render_pass_depth_attachment.view = swapchain_depth;
    render_pass_depth_attachment.depthClearValue = 1.0f;
    render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Clear;
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

    {
        // Create & fill the render pass (encoder)
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

        // Bind Pipeline
        render_quad_pipeline.set(render_pass);

        // Set binding group
        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_group, 0, nullptr);

        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_mesh.get_vertex_buffer(), 0, quad_mesh.get_byte_size());

        // Submit drawcall
        wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

        wgpuRenderPassEncoderEnd(render_pass);

        wgpuRenderPassEncoderRelease(render_pass);
    }

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(command_encoder);
}

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::render_mirror()
{
    // Get the current texture in the swapchain
    WGPUTextureView current_texture_view = wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain);
    assert_msg(current_texture_view != NULL, "Error, dont resize the window please!!");

    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    encoder_desc.label = "Device command encoder";

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Create & fill the render pass (encoder)
    {
        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};
        render_pass_color_attachment.view = current_texture_view;
        render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.clearValue = WGPUColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);

        WGPURenderPassDescriptor render_pass_descr = {};
        render_pass_descr.colorAttachmentCount = 1;
        render_pass_descr.colorAttachments = &render_pass_color_attachment;

        {
            WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

            // Bind Pipeline
            mirror_pipeline.set(render_pass);

            // Set binding group
            wgpuRenderPassEncoderSetBindGroup(render_pass, 0, swapchain_bind_groups[xr_context.swapchains[0].image_index], 0, nullptr);

            // Set vertex buffer while encoding the render pass
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_mesh.get_vertex_buffer(), 0, quad_mesh.get_byte_size());

            // Submit drawcall
            wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

            wgpuRenderPassEncoderEnd(render_pass);

            wgpuRenderPassEncoderRelease(render_pass);
        }
    }

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    // Submit frame to mirror window
    wgpuSwapChainPresent(webgpu_context.screen_swapchain);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(command_encoder);
    wgpuTextureViewRelease(current_texture_view);
}

#endif

void RoomsRenderer::init_render_quad_pipeline()
{
    render_quad_shader = RendererStorage::get_shader("data/shaders/quad_eye.wgsl");

    init_render_quad_bind_groups();

    quad_mesh.create_quad();

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context.xr_swapchain_format : webgpu_context.swapchain_format;

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

    render_quad_pipeline.create_render(render_quad_shader, color_target, true);
}

void RoomsRenderer::init_render_quad_bind_groups()
{
    for (int i = 0; i < EYE_COUNT; ++i)
    {
        eye_textures[i].create(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_RGBA32Float,
            { webgpu_context.render_width, webgpu_context.render_height, 1 },
            static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding),
            1, nullptr);

        eye_depth_textures[i].create(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_Depth32Float,
            { webgpu_context.render_width, webgpu_context.render_height, 1 },
            WGPUTextureUsage_RenderAttachment,
            1, nullptr);

        if (eye_depth_texture_view[i]) {
            wgpuTextureViewRelease(eye_depth_texture_view[i]);
        }

        if (eye_render_bind_group[i]) {
            wgpuBindGroupRelease(eye_render_bind_group[i]);
        }

        // Generate Texture views of depth buffers
        eye_depth_texture_view[i] = eye_depth_textures[i].get_view();

        // Uniforms
        eye_render_texture_uniform[i].data = eye_textures[i].get_view();
        eye_render_texture_uniform[i].binding = 0;

        std::vector<Uniform*> uniforms = { &eye_render_texture_uniform[i] };
        eye_render_bind_group[i] = webgpu_context.create_bind_group(uniforms, render_quad_shader, 0);
    }
}

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::init_mirror_pipeline()
{
    mirror_shader = RendererStorage::get_shader("data/shaders/quad_mirror.wgsl");

    WGPUTextureFormat swapchain_format = webgpu_context.swapchain_format;

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

    // Generate uniforms from the swapchain
    for (uint8_t i = 0; i < xr_context.swapchains[0].images.size(); i++) {
        Uniform swapchain_uni;

        swapchain_uni.data = xr_context.swapchains[0].images[i].textureView;
        swapchain_uni.binding = 0;

        swapchain_uniforms.push_back(swapchain_uni);
    }

    std::vector<Uniform*> uniforms = { &swapchain_uniforms[0] };

    // Generate bindgroups from the swapchain
    for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
        Uniform swapchain_uni;

        std::vector<Uniform*> uniforms = { &swapchain_uniforms[i] };

        swapchain_bind_groups.push_back(webgpu_context.create_bind_group(uniforms, mirror_shader, 0));
    }

    mirror_pipeline.create_render(mirror_shader, color_target);
}

#endif

void RoomsRenderer::init_camera_bindgroup()
{
    camera_uniform.data = webgpu_context.create_buffer(sizeof(glm::mat4x4), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    camera_uniform.binding = 0;
    camera_uniform.buffer_size = sizeof(glm::mat4x4);
}

void RoomsRenderer::resize_window(int width, int height)
{
    Renderer::resize_window(width, height);

    init_render_quad_bind_groups();
}

Texture* RoomsRenderer::get_eye_texture(eEYE eye)
{
    return &eye_textures[eye];
}
