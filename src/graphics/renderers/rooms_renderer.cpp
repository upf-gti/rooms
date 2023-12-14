#include "rooms_renderer.h"

#ifdef XR_SUPPORT
#include "dawnxr/dawnxr_internal.h"
#endif

#include "spdlog/spdlog.h"

RoomsRenderer::RoomsRenderer() : Renderer()
{
    
}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);
    Shader::set_custom_define("MAX_EDITS_PER_EVALUATION", MAX_EDITS_PER_EVALUATION);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    init_depth_buffers();
    init_camera_bind_group();
    init_ibl_bind_group();

    raymarching_renderer.initialize(use_mirror_screen);
    mesh_renderer.initialize();

#ifdef XR_SUPPORT
    if (is_openxr_available && use_mirror_screen) {
        init_mirror_pipeline();
    }
#endif

    camera = new OrbitCamera();

    camera->set_perspective(glm::radians(45.0f), webgpu_context.render_width / static_cast<float>(webgpu_context.render_height), z_near, z_far);
    camera->look_at(glm::vec3(0.0f, 0.05f, 0.2f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera->set_mouse_sensitivity(0.004f);

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    raymarching_renderer.clean();
    mesh_renderer.clean();

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

    delete camera;
}

void RoomsRenderer::update(float delta_time)
{
    camera->update(delta_time);

    raymarching_renderer.update(delta_time);
    mesh_renderer.update(delta_time);
}

void RoomsRenderer::render()
{
    prepare_instancing();

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
    camera_data.eye = camera->get_eye();
    camera_data.mvp = camera->get_view_projection();
    camera_data.dummy = 0.f;

    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &(camera_data), sizeof(sCameraData));

    WGPUTextureView swapchain_view = wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain);

    mesh_renderer.render(swapchain_view, eye_depth_texture_view[EYE_LEFT]);

#ifndef DISABLE_RAYMARCHER
    raymarching_renderer.set_camera_eye(camera->get_eye());
    raymarching_renderer.render_raymarching_proxy(swapchain_view, eye_depth_texture_view[EYE_LEFT]);
#endif
    
    wgpuTextureViewRelease(swapchain_view);

#ifndef __EMSCRIPTEN__
    wgpuSwapChainPresent(webgpu_context.screen_swapchain);
#endif
}

#if defined(XR_SUPPORT)

void RoomsRenderer::render_xr()
{
    xr_context.init_frame();

    for (uint32_t i = 0; i < xr_context.view_count; ++i) {

        xr_context.acquire_swapchain(i);

        const sSwapchainData& swapchainData = xr_context.swapchains[i];

        WebGPUContext* webgpu_context = get_webgpu_context();
        camera_data.eye = xr_context.per_view_data[i].position;
        camera_data.mvp = xr_context.per_view_data[i].view_projection_matrix;
        camera_data.dummy = 0.f;

        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &camera_data, sizeof(sCameraData));

        mesh_renderer.render(swapchainData.images[swapchainData.image_index].textureView, eye_depth_texture_view[i]);

#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.set_camera_eye(xr_context.per_view_data[i].position);
        raymarching_renderer.render_raymarching_proxy(swapchainData.images[swapchainData.image_index].textureView, eye_depth_texture_view[i]);
#endif

        xr_context.release_swapchain(i);
    }

    xr_context.end_frame();
}
#endif

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::render_mirror()
{
    // Get the current texture in the swapchain
    WGPUTextureView current_texture_view = wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain);
    assert(current_texture_view != NULL);

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

void RoomsRenderer::init_depth_buffers()
{
    for (int i = 0; i < EYE_COUNT; ++i)
    {
        eye_depth_textures[i].create(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_Depth32Float,
            { webgpu_context.render_width, webgpu_context.render_height, 1 },
            WGPUTextureUsage_RenderAttachment,
            1, nullptr);

        if (eye_depth_texture_view[i]) {
            wgpuTextureViewRelease(eye_depth_texture_view[i]);
        }

        // Generate Texture views of depth buffers
        eye_depth_texture_view[i] = eye_depth_textures[i].get_view();
    }
}

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::init_mirror_pipeline()
{
    mirror_shader = RendererStorage::get_shader("data/shaders/quad_mirror.wgsl");

    quad_mesh.create_quad();

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

    mirror_pipeline.create_render(mirror_shader, color_target, { .uses_depth_buffer = false });
}

#endif

void RoomsRenderer::init_camera_bind_group()
{
    camera_uniform.data = webgpu_context.create_buffer(sizeof(sCameraData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    camera_uniform.binding = 0;
    camera_uniform.buffer_size = sizeof(sCameraData);
}

void RoomsRenderer::resize_window(int width, int height)
{
    Renderer::resize_window(width, height);

    init_depth_buffers();
}
