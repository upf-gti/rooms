#include "raymarching_renderer.h"

#ifdef XR_SUPPORT
#include "dawnxr/dawnxr_internal.h"
#endif

#include "framework/input.h"

std::ostream& operator<<(std::ostream& os, const sEdit& edit)
{
    os << "Position: " << edit.position.x << ", " << edit.position.y << ", " << edit.position.z << std::endl;
    os << "Primitive: " << edit.primitive << std::endl;
    os << "Color: " << edit.color.x << ", " << edit.color.y << ", " << edit.color.z << std::endl;
    os << "Operation: " << edit.operation << std::endl;
    os << "Size: " << edit.size.x << ", " << edit.size.y << ", " << edit.size.z << std::endl;
    os << "Radius: " << edit.radius << std::endl;
    return os;
}

RaymarchingRenderer::RaymarchingRenderer() : Renderer()
{
    
}

int RaymarchingRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    init_render_quad_pipeline();
    init_render_mesh_pipelines();
    init_compute_raymarching_pipeline();
    init_compute_merge_pipeline();
    init_initialize_sdf_pipeline();

#ifdef XR_SUPPORT
    if (is_openxr_available && use_mirror_screen) {
        init_mirror_pipeline();
    }
#endif

    compute_raymarching_data.render_width = static_cast<float>(render_width);
    compute_raymarching_data.render_height = static_cast<float>(render_height);

    compute_merge_data.sdf_size = glm::uvec3(SDF_RESOLUTION);

    compute_initialize_sdf();

    return 0;
}

void RaymarchingRenderer::clean()
{
    Renderer::clean();

    // Uniforms
    u_compute_buffer_data.destroy();
    u_compute_texture_left_eye.destroy();
    u_compute_texture_right_eye.destroy();
    u_render_texture_left_eye.destroy();
    u_render_texture_right_eye.destroy();

    // Render pipeline
    wgpuBindGroupRelease(render_bind_group_left_eye);
    wgpuBindGroupRelease(render_bind_group_right_eye);

    // Compute pipeline
    wgpuBindGroupRelease(compute_raymarching_textures_bind_group);
    wgpuBindGroupRelease(compute_raymarching_data_bind_group);
    wgpuBindGroupRelease(initialize_sdf_bind_group);

    // Texture views
    wgpuTextureViewRelease(left_eye_depth_texture_view);
    wgpuTextureViewRelease(right_eye_depth_texture_view);

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    if (is_openxr_available) {
        for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
            swapchain_uniforms[i].destroy();
            wgpuBindGroupRelease(swapchain_bind_groups[i]);
        }
    }
#endif
}

void RaymarchingRenderer::update(float delta_time)
{
    compute_raymarching_data.time += delta_time;

#ifdef XR_SUPPORT
    if (Input::is_key_pressed(GLFW_KEY_A) || Input::get_trigger_value(HAND_RIGHT) > 0.5) {

        sEdit edit;
        edit.operation = OP_SMOOTH_UNION; // random() < 0.5 ? OP_SMOOTH_UNION : OP_SMOOTH_SUBSTRACTION;
        edit.color = glm::vec3(random_f(), random_f(), random_f());
        // edit.position = glm::vec3(0.4 * (random() * 2 - 1), 0.4 * (random() * 2 - 1), 0.4 * (random() * 2 - 1));
        edit.position = Input::get_controller_position(HAND_RIGHT);
        edit.position.y -= 1.f;
        edit.primitive = SD_SPHERE;
        edit.size = glm::vec3(1.0, 1.0, 1.0);
        edit.radius = 0.02f;// random();

        std::cout << edit << std::endl;

        push_edit(edit);
    }

    if (Input::get_trigger_value(HAND_LEFT) > 0.5) {

        sEdit edit;
        edit.operation = OP_SMOOTH_SUBSTRACTION;
        edit.color = glm::vec3(random_f(), random_f(), random_f());
        // edit.position = glm::vec3(0.4 * (random() * 2 - 1), 0.4 * (random() * 2 - 1), 0.4 * (random() * 2 - 1));
        edit.position = Input::get_controller_position(HAND_LEFT);
        edit.position.y -= 1.f;
        edit.primitive = SD_SPHERE;
        edit.size = glm::vec3(1.0, 1.0, 1.0);
        edit.radius = 0.02f;// random();

        //std::cout << edit << std::endl;

        push_edit(edit);
    }
#else
    if (Input::is_key_pressed(GLFW_KEY_A)) {

        sEdit edit;
        edit.operation = OP_SMOOTH_UNION; // random() < 0.5 ? OP_SMOOTH_UNION : OP_SMOOTH_SUBSTRACTION;
        edit.color = glm::vec3(random_f(), random_f(), random_f());
        edit.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
        //edit.position.z += 0.20f;
        edit.primitive = SD_SPHERE;
        edit.size = glm::vec3(1.0, 1.0, 1.0);
        edit.radius = 0.02f;// random();

        //std::cout << edit << std::endl;

        //push_edit(edit);
    }
#endif
}

void RaymarchingRenderer::render()
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

    render_mesh_pipeline.clean_renderables();
    render_mesh_texture_pipeline.clean_renderables();
    render_fonts_pipeline.clean_renderables();

    // Check validation errors
    webgpu_context.print_errors();
}

void RaymarchingRenderer::render_screen()
{
    glm::vec3 eye = glm::vec3(0.0f, 2.0f, 1.5f);
    glm::mat4x4 view = glm::lookAt(eye, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 projection = glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;

    glm::mat4x4 view_projection = projection * view;

    compute_raymarching_data.inv_view_projection_left_eye = glm::inverse(view_projection);
    compute_raymarching_data.left_eye_pos = eye;
    compute_raymarching_data.camera_far = 100.0f;
    compute_raymarching_data.camera_near = 0.1f;

    camera_data.view_projection = view_projection;

    // Update uniform buffer
    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_camera.data), 0, &(camera_data), sizeof(sCameraData));

    compute_merge();
    compute_raymarching();

    WGPUTextureView swapchain_view = wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain);

    render_eye_quad(swapchain_view, left_eye_depth_texture_view, render_bind_group_left_eye);

    render_meshes(swapchain_view, left_eye_depth_texture_view);
    
    wgpuTextureViewRelease(swapchain_view);

#ifndef __EMSCRIPTEN__
    wgpuSwapChainPresent(webgpu_context.screen_swapchain);
#endif
}

void RaymarchingRenderer::render_meshes(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth)
{
    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Prepare the color attachment
    WGPURenderPassColorAttachment render_pass_color_attachment = {};
    render_pass_color_attachment.view = swapchain_view;
    render_pass_color_attachment.loadOp = WGPULoadOp_Load;
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
    render_pass_color_attachment.clearValue = WGPUColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Prepate the depth attachment
    WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
    render_pass_depth_attachment.view = swapchain_depth;
    render_pass_depth_attachment.depthClearValue = 1.0f;
    render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Load;
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

    static auto render_pipeline = [&](Pipeline& pipeline) {

        // Bind Pipeline
        pipeline.set(render_pass);

        for (const auto entity : pipeline.get_render_list()) {

            Mesh* mesh = entity->get_mesh();

            // Not initialized
            if (mesh->get_vertex_count() == 0) {
                std::cerr << "Skipping not initialized mesh" << std::endl;
                continue;
            }

            // Set bind group
            wgpuRenderPassEncoderSetBindGroup(render_pass, 0, mesh->get_bind_group(), 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(render_pass, 1, render_bind_group_camera, 0, nullptr);

            // Set vertex buffer while encoding the render pass
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, mesh->get_vertex_buffer(), 0, mesh->get_byte_size());

            // Submit drawcall
            wgpuRenderPassEncoderDraw(render_pass, static_cast<uint32_t>(mesh->get_vertex_count()), 1, 0, 0);
        }
    };

    render_pipeline(render_mesh_pipeline);
    render_pipeline(render_mesh_texture_pipeline);
    render_pipeline(render_fonts_pipeline);

    wgpuRenderPassEncoderEnd(render_pass);

    wgpuRenderPassEncoderRelease(render_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(command_encoder);
}

#if defined(XR_SUPPORT)

void RaymarchingRenderer::render_xr()
{
    if (is_openxr_available) {

        xr_context.init_frame();

        compute_raymarching_data.inv_view_projection_left_eye = glm::inverse(xr_context.per_view_data[0].view_projection_matrix);
        compute_raymarching_data.inv_view_projection_right_eye = glm::inverse(xr_context.per_view_data[1].view_projection_matrix);

        compute_raymarching_data.left_eye_pos = xr_context.per_view_data[0].position;
        compute_raymarching_data.right_eye_pos = xr_context.per_view_data[1].position;

        const float* proj_verts = glm::value_ptr(xr_context.per_view_data[0].projection_matrix);

        compute_raymarching_data.camera_far = proj_verts[14] / (proj_verts[10] - 1.0);
        compute_raymarching_data.camera_near = proj_verts[14] / (proj_verts[10] + 1.0);

        compute_merge();
        compute_raymarching();

        for (uint32_t i = 0; i < xr_context.view_count; ++i) {

            xr_context.acquire_swapchain(i);

            const sSwapchainData& swapchainData = xr_context.swapchains[i];

            WGPUBindGroup bind_group = i == 0 ? render_bind_group_left_eye : render_bind_group_right_eye;
            WGPUTextureView depth_texture_view = (i == 0) ? left_eye_depth_texture_view : right_eye_depth_texture_view;

            render_eye_quad(swapchainData.images[swapchainData.image_index].textureView, depth_texture_view, bind_group);

            camera_data.view_projection = xr_context.per_view_data[i].view_projection_matrix;

            // Update uniform buffer
            wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_camera.data), 0, &(camera_data), sizeof(sCameraData));

            render_meshes(swapchainData.images[swapchainData.image_index].textureView, depth_texture_view);

            xr_context.release_swapchain(i);
        }

        xr_context.end_frame();
    }
    else {
        render_screen();
    }
}
#endif

void RaymarchingRenderer::render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group)
{
    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Prepare the color attachment
    WGPURenderPassColorAttachment render_pass_color_attachment = {};
    render_pass_color_attachment.view = swapchain_view;
    render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
    render_pass_color_attachment.clearValue = WGPUColor(0.0f, 0.0f, 0.0f, 1.0f);

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

void RaymarchingRenderer::compute_initialize_sdf() {
    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWriteCount = 0;
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute_raymarching pass
    initialize_sdf_pipeline.set(compute_pass);

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, initialize_sdf_bind_group, 0, nullptr);

    uint32_t workgroupSize = 8;
    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupHeight = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupDepth = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, workgroupDepth);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Initialize SDF Command buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);
}

void RaymarchingRenderer::compute_merge()
{
    // Nothing to merge if equals 0
    if (compute_merge_data.edits_to_process == 0) {
        return;
    }

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWriteCount = 0;
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute_raymarching pass
    compute_merge_pipeline.set(compute_pass);

    // Update uniform buffer
    std::cout << compute_merge_data.edits_to_process << std::endl;
    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_compute_edits_array.data), 0, edits, sizeof(sEdit) * compute_merge_data.edits_to_process);
    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_compute_merge_data.data), 0, &(compute_merge_data), sizeof(sMergeData));

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_merge_bind_group, 0, nullptr);

    uint32_t workgroupSize = 8;
    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupHeight = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupDepth = (SDF_RESOLUTION + workgroupSize - 1) / workgroupSize;
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, workgroupDepth);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Merge Command buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);

    compute_merge_data.edits_to_process = 0;

}

void RaymarchingRenderer::compute_raymarching()
{
    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWriteCount = 0;
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute_raymarching pass
    compute_raymarching_pipeline.set(compute_pass);

    // Update uniform buffer
    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_compute_buffer_data.data), 0, &(compute_raymarching_data), sizeof(sComputeData));

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_raymarching_textures_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_raymarching_data_bind_group, 0, nullptr);

    uint32_t workgroupSize = 16;
    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth = (render_width + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupHeight = (render_height + workgroupSize - 1) / workgroupSize;
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, 1);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Raymarch Command buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);
}

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RaymarchingRenderer::render_mirror()
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
        render_pass_color_attachment.clearValue = WGPUColor(0.0f, 0.0f, 0.0f, 1.0f);

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
        }
    }

    //
    {
        WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
        cmd_buff_descriptor.nextInChain = NULL;
        cmd_buff_descriptor.label = "Command buffer";

        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
        wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

        //commander.Release();
        //current_texture_view.Release();
    }

    // Submit frame to mirror window
    {
        wgpuSwapChainPresent(webgpu_context.screen_swapchain);
    }
}

#endif

void RaymarchingRenderer::init_render_quad_pipeline()
{
    render_quad_shader = Shader::get("data/shaders/quad_eye.wgsl");

    left_eye_texture.create(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_RGBA8Unorm,
        { render_width, render_height, 1 },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding),
        1, nullptr);

    left_eye_depth_texture.create(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_Depth16Unorm,
        { render_width, render_height, 1 },
        WGPUTextureUsage_RenderAttachment,
        1, nullptr);

    right_eye_texture.create(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_RGBA8Unorm,
        { render_width, render_height, 1 },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding),
        1, nullptr);

    right_eye_depth_texture.create(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_Depth16Unorm,
        { render_width, render_height, 1 },
        WGPUTextureUsage_RenderAttachment,
        1, nullptr);

    // Generate Texture views of depth buffers
    left_eye_depth_texture_view = left_eye_depth_texture.get_view();
    right_eye_depth_texture_view = right_eye_depth_texture.get_view();

    // Generate bindgroups
    u_render_texture_left_eye.data = left_eye_texture.get_view();
    u_render_texture_left_eye.binding = 0;

    u_render_texture_right_eye.data = right_eye_texture.get_view();
    u_render_texture_right_eye.binding = 0;

    // Left eye bind group
    {
        std::vector<Uniform*> uniforms = { &u_render_texture_left_eye };

        render_bind_group_left_eye = webgpu_context.create_bind_group(uniforms, render_quad_shader, 0);
    }

    // Right eye bind group
    {
        std::vector<Uniform*> uniforms = { &u_render_texture_right_eye };

        render_bind_group_right_eye = webgpu_context.create_bind_group(uniforms, render_quad_shader, 0);
    }

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

void RaymarchingRenderer::init_render_mesh_pipelines()
{
    render_mesh_shader = Shader::get("data/shaders/mesh_color.wgsl");
    render_mesh_texture_shader = Shader::get("data/shaders/mesh_texture.wgsl");
    render_fonts_shader = Shader::get("data/shaders/sdf_fonts.wgsl");

    u_camera.data = webgpu_context.create_buffer(sizeof(sCameraData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    u_camera.binding = 0;
    u_camera.buffer_size = sizeof(sCameraData);

    std::vector<Uniform*> uniforms = { &u_camera };

    render_bind_group_camera = webgpu_context.create_bind_group(uniforms, render_mesh_shader, 1);

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

    render_mesh_pipeline.create_render(render_mesh_shader, color_target, true);
    render_mesh_texture_pipeline.create_render(render_mesh_texture_shader, color_target, true);
    render_fonts_pipeline.create_render(render_fonts_shader, color_target, true);
}

void RaymarchingRenderer::init_compute_raymarching_pipeline()
{
    // Load compute_raymarching shader
    compute_raymarching_shader = Shader::get("data/shaders/sdf_raymarching.wgsl");

    // Texture uniforms
    {
        u_compute_texture_left_eye.data = left_eye_texture.get_view();
        u_compute_texture_left_eye.binding = 0;

        u_compute_texture_right_eye.data = right_eye_texture.get_view();
        u_compute_texture_right_eye.binding = 1;

        // RGB for color, A for distance
        
        //std::vector<glm::vec4> sdf_data = { SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION, glm::vec4(0.0, 0.0, 0.0, 1000.0) };

        u_compute_texture_sdf_storage.data = webgpu_context.create_buffer(SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr);
        u_compute_texture_sdf_storage.binding = 2;
        u_compute_texture_sdf_storage.buffer_size = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4;

        std::vector<Uniform*> uniforms = { &u_compute_texture_left_eye, &u_compute_texture_right_eye, &u_compute_texture_sdf_storage };

        compute_raymarching_textures_bind_group = webgpu_context.create_bind_group(uniforms, compute_raymarching_shader, 0);
    }

    // Compute data uniforms
    {
        u_compute_buffer_data.data = webgpu_context.create_buffer(sizeof(sComputeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        u_compute_buffer_data.binding = 0;
        u_compute_buffer_data.buffer_size = sizeof(sComputeData);

        std::vector<Uniform*> uniforms = { &u_compute_buffer_data };

        compute_raymarching_data_bind_group = webgpu_context.create_bind_group(uniforms, compute_raymarching_shader, 1);
    }

    compute_raymarching_pipeline.create_compute(compute_raymarching_shader);
}

void RaymarchingRenderer::init_initialize_sdf_pipeline() {
    initialize_sdf_shader = Shader::get("data/shaders/sdf_initialization.wgsl");

    std::vector<Uniform*> uniforms = { &u_compute_texture_sdf_storage };
    initialize_sdf_bind_group = webgpu_context.create_bind_group(uniforms, initialize_sdf_shader, 0);
    
    initialize_sdf_pipeline.create_compute(initialize_sdf_shader);
}

void RaymarchingRenderer::init_compute_merge_pipeline()
{
    // Load compute_raymarching shader
    compute_merge_shader = Shader::get("data/shaders/sdf_merge.wgsl");

    // Texture uniforms
    {
        u_compute_edits_array.data = webgpu_context.create_buffer(sizeof(edits), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        u_compute_edits_array.binding = 0;
        u_compute_edits_array.buffer_size = sizeof(edits);

        u_compute_merge_data.data = webgpu_context.create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        u_compute_merge_data.binding = 1;
        u_compute_merge_data.buffer_size = sizeof(sMergeData);

        std::vector<Uniform*> uniforms = { &u_compute_edits_array, &u_compute_merge_data, &u_compute_texture_sdf_storage };

        compute_merge_bind_group = webgpu_context.create_bind_group(uniforms, compute_merge_shader, 0);
    }

    compute_merge_pipeline.create_compute(compute_merge_shader);
}

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)

void RaymarchingRenderer::init_mirror_pipeline()
{
    mirror_shader = Shader::get("data/shaders/quad_mirror.wgsl");

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
        swapchain_uni.visibility = WGPUShaderStage_Fragment;

        swapchain_uniforms.push_back(swapchain_uni);
    }

    WGPUBindGroupLayout swapchain_bind_group_layout;
    std::vector<Uniform*> uniforms = { &swapchain_uniforms[0] };
    swapchain_bind_group_layout = webgpu_context.create_bind_group_layout(uniforms);
    WGPUPipelineLayout render_mirror_pipeline_layout = webgpu_context.create_pipeline_layout({ swapchain_bind_group_layout });

    // Generate bindgroups from the swapchain
    for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
        Uniform swapchain_uni;

        std::vector<Uniform*> uniforms = { &swapchain_uniforms[i] };

        swapchain_bind_groups.push_back(webgpu_context.create_bind_group(uniforms, swapchain_bind_group_layout));
    }

    mirror_pipeline.create_render(mirror_shader, color_target);
}

#endif