#include "rooms_renderer.h"
#include "graphics/debug/renderdoc_capture.h"
#include "graphics/shader.h"
#include "graphics/renderer_storage.h"

#include "framework/camera/camera_2d.h"
#include "framework/input.h"
#include "framework/ui/io.h"

#ifdef XR_SUPPORT
#include "xr/openxr_context.h"
#include "xr/dawnxr/dawnxr_internal.h"
#endif

#include "spdlog/spdlog.h"

#include "shaders/mesh_pbr.wgsl.gen.h"
#include "shaders/mesh_color.wgsl.gen.h"
#include "shaders/quad_mirror.wgsl.gen.h"

#include "backends/imgui_impl_wgpu.h"

RoomsRenderer::RoomsRenderer() : Renderer()
{
    
}

RoomsRenderer::~RoomsRenderer()
{

}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    size_t s = sizeof(Stroke) * 400;

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);
    Shader::set_custom_define("MAX_EDITS_PER_EVALUATION", MAX_EDITS_PER_EVALUATION);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    init_camera_bind_group();

    raymarching_renderer.initialize(use_mirror_screen);

#ifdef XR_SUPPORT
    if (is_openxr_available && use_mirror_screen) {
        init_mirror_pipeline();
    }
#endif

    // Main 3D Camera

    camera = new FlyoverCamera();
    camera->set_perspective(glm::radians(45.0f), webgpu_context->render_width / static_cast<float>(webgpu_context->render_height), z_near, z_far);
    camera->look_at(glm::vec3(0.0f, 0.1f, 0.4f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera->set_mouse_sensitivity(0.003f);
    camera->set_speed(0.5f);

    std::vector<Uniform*> uniforms = { &camera_uniform };
    render_bind_group_camera = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_pbr::source, shaders::mesh_pbr::path), 1);

    // Orthographic camera for ui rendering

    float w = static_cast<float>(webgpu_context->render_width);
    float h = static_cast<float>(webgpu_context->render_height);

    camera_2d = new Camera2D();
    camera_2d->set_orthographic(0.0f, w, 0.0f, h, -1.0f, 1.0f);

    uniforms = { &camera_2d_uniform };
    render_bind_group_camera_2d = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_color::source, shaders::mesh_color::path), 1);

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    raymarching_renderer.clean();

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
#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        xr_context->update();
    }
#endif

    if (debug_this_frame) {
        RenderdocCapture::start_capture_frame();
    }
    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    global_command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    if (!is_openxr_available) {
        const auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse && !io.WantCaptureKeyboard && !IO::any_focus()) {
            camera->update(delta_time);
        }
    } else {
        camera->update(delta_time);
    }

    raymarching_renderer.update_sculpt(global_command_encoder);
}

void RoomsRenderer::render()
{
    prepare_instancing();

    WGPUTextureView swapchain_view;

    if (!is_openxr_available || use_mirror_screen) {
        swapchain_view = wgpuSwapChainGetCurrentTextureView(webgpu_context->screen_swapchain);
    }

    if (!is_openxr_available) {
        render_screen(swapchain_view);
    }

#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        render_xr();

        if (use_mirror_screen) {
            render_mirror(swapchain_view);
        }
    }
#endif

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    resolve_query_set(global_command_encoder, 0);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(global_command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(global_command_encoder);

    if (debug_this_frame) {
        RenderdocCapture::end_capture_frame();
        debug_this_frame = false;
    }

    if (!is_openxr_available) {
        wgpuTextureViewRelease(swapchain_view);
    }
#ifdef XR_SUPPORT
    else {
        xr_context->end_frame();
    }
#endif

#ifndef __EMSCRIPTEN__
    if (!is_openxr_available || use_mirror_screen) {
        wgpuSwapChainPresent(webgpu_context->screen_swapchain);
    }
#endif

    last_frame_timestamps = get_timestamps();

    if (!last_frame_timestamps.empty() && raymarching_renderer.has_performed_evaluation()) {
        last_evaluation_time = last_frame_timestamps[0];
    }

    clear_renderables();
}

void RoomsRenderer::render_screen(WGPUTextureView swapchain_view)
{
    // Update main 3d camera

    camera_data.eye = camera->get_eye();
    camera_data.mvp = camera->get_view_projection();

    // Use camera position as controller position
    camera_data.right_controller_position = camera_data.eye;

    camera_data.exposure = exposure;
    camera_data.ibl_intensity = ibl_intensity;

    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &camera_data, sizeof(sCameraData));

    // Update 2d camera for UI

    camera_2d_data.eye = camera_2d->get_eye();
    camera_2d_data.mvp = camera_2d->get_view_projection();

    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_2d_uniform.data), 0, &camera_2d_data, sizeof(sCameraData));

    ImGui::Render();

    {
        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};
        if (msaa_count > 1) {
            render_pass_color_attachment.view = multisample_textures_views[0];
            render_pass_color_attachment.resolveTarget = swapchain_view;
        }
        else {
            render_pass_color_attachment.view = swapchain_view;
        }

        render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
        render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

        // Prepate the depth attachment
        WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
        render_pass_depth_attachment.view = eye_depth_texture_view[EYE_LEFT];
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

        std::vector<WGPURenderPassTimestampWrites> timestampWrites(1);
        timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, "pre_render");
        timestampWrites[0].querySet = timestamp_query_set;
        timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, "render");

        render_pass_descr.timestampWrites = timestampWrites.data();

        // Create & fill the render pass (encoder)
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

        render_opaque(render_pass, render_bind_group_camera);

#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.render_raymarching_proxy(render_pass);
#endif

        render_transparent(render_pass, render_bind_group_camera);

        render_2D(render_pass, render_bind_group_camera_2d);

        wgpuRenderPassEncoderEnd(render_pass);

        wgpuRenderPassEncoderRelease(render_pass);

        //timestamp(global_command_encoder, "render");

        // render imgui
        {
            WGPURenderPassColorAttachment color_attachments = {};
            color_attachments.view = swapchain_view;
            color_attachments.loadOp = WGPULoadOp_Load;
            color_attachments.storeOp = WGPUStoreOp_Store;
            color_attachments.clearValue = { 0.0, 0.0, 0.0, 0.0 };
            color_attachments.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

            WGPURenderPassDescriptor render_pass_desc = {};
            render_pass_desc.colorAttachmentCount = 1;
            render_pass_desc.colorAttachments = &color_attachments;
            render_pass_desc.depthStencilAttachment = nullptr;

            WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_desc);

            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
        }
    }
}

#if defined(XR_SUPPORT)

void RoomsRenderer::render_xr()
{
    xr_context->init_frame();

    for (uint32_t i = 0; i < xr_context->view_count; ++i) {

        xr_context->acquire_swapchain(i);

        const sSwapchainData& swapchainData = xr_context->swapchains[i];

        camera_data.eye = xr_context->per_view_data[i].position;
        camera_data.mvp = xr_context->per_view_data[i].view_projection_matrix;

        camera_data.right_controller_position = Input::get_controller_position(HAND_RIGHT);

        camera_data.exposure = exposure;
        camera_data.ibl_intensity = ibl_intensity;

        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), i * camera_buffer_stride, &camera_data, sizeof(sCameraData));

        {
            // Prepare the color attachment
            WGPURenderPassColorAttachment render_pass_color_attachment = {};
            render_pass_color_attachment.view = swapchainData.images[swapchainData.image_index].textureView;

            if (msaa_count > 1) {
                render_pass_color_attachment.view = multisample_textures_views[i];
                render_pass_color_attachment.resolveTarget = swapchainData.images[swapchainData.image_index].textureView;
            }
            else {
                render_pass_color_attachment.view = swapchainData.images[swapchainData.image_index].textureView;
            }

            render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
            render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
            render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

            glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
            render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

            // Prepate the depth attachment
            WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
            render_pass_depth_attachment.view = eye_depth_texture_view[i];
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

            // Create & fill the render pass (encoder)
            WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

            render_opaque(render_pass, render_bind_group_camera, i * camera_buffer_stride);

#ifndef DISABLE_RAYMARCHER
            raymarching_renderer.render_raymarching_proxy(render_pass, i * camera_buffer_stride);
#endif

            render_transparent(render_pass, render_bind_group_camera, i * camera_buffer_stride);

            render_2D(render_pass, render_bind_group_camera_2d);

            wgpuRenderPassEncoderEnd(render_pass);
            wgpuRenderPassEncoderRelease(render_pass);
        }

        xr_context->release_swapchain(i);
    }
}
#endif

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::render_mirror(WGPUTextureView swapchain_view)
{
    ImGui::Render();

    // Create & fill the render pass (encoder)
    {
        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};
        render_pass_color_attachment.view = swapchain_view;
        render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.clearValue = WGPUColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
        render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        WGPURenderPassDescriptor render_pass_descr = {};
        render_pass_descr.colorAttachmentCount = 1;
        render_pass_descr.colorAttachments = &render_pass_color_attachment;
        render_pass_descr.depthStencilAttachment = nullptr;

        {
            WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

            // Bind Pipeline
            mirror_pipeline.set(render_pass);

            // Set binding group
            wgpuRenderPassEncoderSetBindGroup(render_pass, 0, swapchain_bind_groups[xr_context->swapchains[0].image_index], 0, nullptr);

            // Set vertex buffer while encoding the render pass
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_surface.get_vertex_buffer(), 0, quad_surface.get_byte_size());

            // Submit drawcall
            wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

            wgpuRenderPassEncoderEnd(render_pass);

            wgpuRenderPassEncoderRelease(render_pass);
        }
    }

    // render imgui
    {
        WGPURenderPassColorAttachment color_attachments = {};
        color_attachments.view = swapchain_view;
        color_attachments.loadOp = WGPULoadOp_Load;
        color_attachments.storeOp = WGPUStoreOp_Store;
        color_attachments.clearValue = { 0.0, 0.0, 0.0, 0.0 };
        color_attachments.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        WGPURenderPassDescriptor render_pass_desc = {};
        render_pass_desc.colorAttachmentCount = 1;
        render_pass_desc.colorAttachments = &color_attachments;
        render_pass_desc.depthStencilAttachment = nullptr;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_desc);

        ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }
}

#endif

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RoomsRenderer::init_mirror_pipeline()
{
    mirror_shader = RendererStorage::get_shader_from_source(shaders::quad_mirror::source, shaders::quad_mirror::path);

    quad_surface.create_quad(2.0f, 2.0f);

    WGPUTextureFormat swapchain_format = webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    // Generate uniforms from the swapchain
    for (uint8_t i = 0; i < xr_context->swapchains[0].images.size(); i++) {
        Uniform swapchain_uni;

        swapchain_uni.data = xr_context->swapchains[0].images[i].textureView;
        swapchain_uni.binding = 0;

        swapchain_uniforms.push_back(swapchain_uni);
    }

    linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
    linear_sampler_uniform.binding = 1;

    // Generate bindgroups from the swapchain
    for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
        Uniform swapchain_uni;

        std::vector<Uniform*> uniforms = { &swapchain_uniforms[i], &linear_sampler_uniform };

        swapchain_bind_groups.push_back(webgpu_context->create_bind_group(uniforms, mirror_shader, 0));
    }

    mirror_pipeline.create_render(mirror_shader, color_target, { .use_depth = false });
}

#endif

void RoomsRenderer::init_camera_bind_group()
{
    camera_buffer_stride = std::max(static_cast<uint32_t>(sizeof(sCameraData)), required_limits.limits.minUniformBufferOffsetAlignment);

    camera_uniform.data = webgpu_context->create_buffer(camera_buffer_stride * EYE_COUNT, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    camera_uniform.binding = 0;
    camera_uniform.buffer_size = sizeof(sCameraData);

    camera_2d_uniform.data = webgpu_context->create_buffer(sizeof(sCameraData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_2d_buffer");
    camera_2d_uniform.binding = 0;
    camera_2d_uniform.buffer_size = sizeof(sCameraData);
}

void RoomsRenderer::resize_window(int width, int height)
{
    Renderer::resize_window(width, height);
}

glm::vec3 RoomsRenderer::get_camera_eye()
{
#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        return xr_context->per_view_data[0].position; // return left eye
    }
#endif

    return camera->get_eye();
}

glm::vec3 RoomsRenderer::get_camera_front()
{
#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        glm::mat4x4 view = xr_context->per_view_data[0].view_matrix; // use left eye
        return { view[2].x, view[2].y, -view[2].z };
    }
#endif

    Camera* camera = get_camera();
    return glm::normalize(camera->get_center() - camera->get_eye());
}
