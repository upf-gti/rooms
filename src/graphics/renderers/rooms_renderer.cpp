#include "rooms_renderer.h"
#include "graphics/debug/renderdoc_capture.h"
#include "graphics/shader.h"
#include "graphics/renderer_storage.h"

#include "framework/input.h"
#include "framework/ui/io.h"

#include "framework/camera/flyover_camera.h"
#include "framework/camera/orbit_camera.h"

#ifdef XR_SUPPORT
#include "xr/openxr_context.h"
#include "xr/dawnxr/dawnxr_internal.h"
#endif

#include "spdlog/spdlog.h"

#include "shaders/mesh_forward.wgsl.gen.h"

RoomsRenderer::RoomsRenderer() : Renderer()
{

}

RoomsRenderer::~RoomsRenderer()
{

}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    raymarching_renderer.initialize(use_mirror_screen);

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    raymarching_renderer.clean();
}

void RoomsRenderer::update(float delta_time)
{
    if (debug_this_frame) {
        RenderdocCapture::start_capture_frame();
    }

    Renderer::update(delta_time);

    raymarching_renderer.update_sculpt(global_command_encoder);
}

void RoomsRenderer::render()
{
    //if (selected_mesh_aabb) {

    //    AABB aabb_transformed = aabb.transform(get_global_model());

    //    selected_mesh_aabb->set_position(aabb_transformed.center);
    //    selected_mesh_aabb->set_scale(aabb_transformed.half_size * 2.0f);

    //    selected_mesh_aabb->render();
    //}

    glm::vec3 camera_position;

    if (!is_openxr_available) {
        if (!frustum_camera_paused) {
            frustum_cull.set_view_projection(camera->get_view_projection());
        }

        camera_position = camera->get_eye();
    }
    else {
        // TODO: use both eyes, only left eye for now
        frustum_cull.set_view_projection(xr_context->per_view_data[0].view_projection_matrix);
        camera_position = xr_context->per_view_data[0].position;
    }

    prepare_instancing(camera_position);

    WGPUTextureView screen_surface_texture_view;
    WGPUSurfaceTexture screen_surface_texture;

    if (!is_openxr_available || use_mirror_screen) {

        wgpuSurfaceGetCurrentTexture(webgpu_context->surface, &screen_surface_texture);
        if (screen_surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_Success) {
            spdlog::error("Error getting swapchain texture");
            return;
        }

        screen_surface_texture_view = webgpu_context->create_texture_view(screen_surface_texture.texture, WGPUTextureViewDimension_2D, webgpu_context->swapchain_format);
    }

    if (!is_openxr_available) {
        render_screen(screen_surface_texture_view);
    }

#if defined(XR_SUPPORT)
    if (is_openxr_available) {
        render_xr();

        if (use_mirror_screen) {
            render_mirror(screen_surface_texture_view);
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

    if (RenderdocCapture::is_capture_started() && debug_this_frame) {
        RenderdocCapture::end_capture_frame();
        debug_this_frame = false;
    }

    if (!is_openxr_available) {
        wgpuTextureViewRelease(screen_surface_texture_view);
        wgpuTextureRelease(screen_surface_texture.texture);
    }

#ifdef XR_SUPPORT
    else {
        xr_context->end_frame();
    }
#endif

#ifndef __EMSCRIPTEN__
    if (!is_openxr_available || use_mirror_screen) {
        wgpuSurfacePresent(webgpu_context->surface);
    }
#endif

    last_frame_timestamps = get_timestamps();

    if (!last_frame_timestamps.empty() && raymarching_renderer.has_performed_evaluation()) {
        last_evaluation_time = last_frame_timestamps[0];
    }

    clear_renderables();
}

void RoomsRenderer::render_screen(WGPUTextureView screen_surface_texture_view)
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
            render_pass_color_attachment.resolveTarget = screen_surface_texture_view;
        }
        else {
            render_pass_color_attachment.view = screen_surface_texture_view;
        }

        render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
        render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

        // Prepate the depth attachment
        WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
        render_pass_depth_attachment.view = eye_depth_texture_view[EYE_LEFT];
        render_pass_depth_attachment.depthClearValue = 0.0f;
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
            color_attachments.view = screen_surface_texture_view;
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
            render_pass_depth_attachment.depthClearValue = 0.0f;
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
