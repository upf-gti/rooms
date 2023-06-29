#include "raymarching_renderer.h"

#ifdef XR_SUPPORT
#include "dawnxr/dawnxr_internal.h"
#endif

RaymarchingRenderer::RaymarchingRenderer() : Renderer()
{
}

int RaymarchingRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    init_render_pipeline();
    init_compute_raymarching_pipeline();
    init_compute_merge_pipeline();

#ifdef XR_SUPPORT
    if (is_openxr_available && use_mirror_screen) {
        init_mirror_pipeline();
    }
#endif

    compute_data.render_width = static_cast<float>(render_width);
    compute_data.render_height = static_cast<float>(render_height);

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
    wgpuRenderPipelineRelease(render_pipeline);
    wgpuPipelineLayoutRelease(render_pipeline_layout);
    wgpuBindGroupRelease(render_bind_group_left_eye);
    wgpuBindGroupRelease(render_bind_group_right_eye);

    // Compute pipeline
    wgpuComputePipelineRelease(compute_raymarching_pipeline);
    wgpuBindGroupRelease(compute_raymarching_textures_bind_group);
    wgpuBindGroupRelease(compute_raymarching_data_bind_group);

    wgpuTextureDestroy(left_eye_texture);
    wgpuTextureDestroy(right_eye_texture);

    // Mesh
    quad_mesh.destroy();

    wgpuBufferDestroy(quad_vertex_buffer);

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    if (is_openxr_available) {
        wgpuRenderPipelineRelease(mirror_pipeline);
    }
#endif
}

void RaymarchingRenderer::update(float delta_time)
{
    compute_data.time += delta_time;
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
}

void RaymarchingRenderer::render_screen()
{
    glm::mat4x4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 projection = glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;

    glm::mat4x4 view_projection = projection * view;

    compute_data.inv_view_projection_left_eye = glm::inverse(view_projection);
    compute_data.left_eye_pos = glm::vec3(0.0, 0.0, 2.0);

    compute();

    render(wgpuSwapChainGetCurrentTextureView(webgpu_context.screen_swapchain), render_bind_group_left_eye);

#ifndef __EMSCRIPTEN__
    wgpuSwapChainPresent(webgpu_context.screen_swapchain);
#endif
}

#if defined(XR_SUPPORT)

void RaymarchingRenderer::render_xr()
{
    if (is_openxr_available) {

        xr_context.init_frame();

        compute_data.inv_view_projection_left_eye = glm::inverse(xr_context.per_view_data[0].view_projection_matrix);
        compute_data.inv_view_projection_right_eye = glm::inverse(xr_context.per_view_data[1].view_projection_matrix);

        compute_data.left_eye_pos = xr_context.per_view_data[0].position;
        compute_data.right_eye_pos = xr_context.per_view_data[1].position;

        compute();

        for (uint32_t i = 0; i < xr_context.view_count; ++i) {

            xr_context.acquire_swapchain(i);

            const sSwapchainData& swapchainData = xr_context.swapchains[i];

            WGPUBindGroup bind_group = i == 0 ? render_bind_group_left_eye : render_bind_group_right_eye;

            render(swapchainData.images[swapchainData.image_index].textureView, bind_group);

            xr_context.release_swapchain(i);
        }

        xr_context.end_frame();
    }
    else {
        render_screen();
    }
}
#endif

void RaymarchingRenderer::render(WGPUTextureView swapchain_view, WGPUBindGroup bind_group)
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

    WGPURenderPassDescriptor render_pass_descr = {};
    render_pass_descr.colorAttachmentCount = 1;
    render_pass_descr.colorAttachments = &render_pass_color_attachment;

    {
        // Create & fill the render pass (encoder)
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

        // Bind Pipeline
        wgpuRenderPassEncoderSetPipeline(render_pass, render_pipeline);

        // Set binding group
        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_group, 0, nullptr);

        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_vertex_buffer, 0, quad_mesh.get_size() * sizeof(float));

        // Submit drawcall
        wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

        wgpuRenderPassEncoderEnd(render_pass);
        //render_pass.Release();
    }

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);
    //webgpu_context.device_command_encoder.Release();

    // Check validation errors
    webgpu_context.printErrors();

}

void RaymarchingRenderer::compute()
{
    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context.device, &encoder_desc);

    // Create compute pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWriteCount = 0;
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute pass
    wgpuComputePassEncoderSetPipeline(compute_pass, compute_raymarching_pipeline);

    // Update uniform buffer
    wgpuQueueWriteBuffer(webgpu_context.device_queue, std::get<WGPUBuffer>(u_compute_buffer_data.data), 0, &(compute_data), sizeof(sComputeData));

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_raymarching_textures_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_raymarching_data_bind_group, 0, nullptr);

    uint32_t workgroupSize = 16;
    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth = (render_width + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupHeight = (render_height + workgroupSize - 1) / workgroupSize;
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, 1);

    // Finalize compute pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context.device_queue, 1, &commands);

    // Check validation errors
    webgpu_context.printErrors();
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
            wgpuRenderPassEncoderSetPipeline(render_pass, mirror_pipeline);

            // Set binding group
            wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_bind_group_left_eye, 0, nullptr);

            // Set vertex buffer while encoding the render pass
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_vertex_buffer, 0, quad_mesh.get_size() * sizeof(float));

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

    // Check validation errors
    webgpu_context.printErrors();
}

#endif

void RaymarchingRenderer::init_render_pipeline()
{
    if (is_openxr_available) {
        render_shader = Shader::get("data/shaders/quad_eye.wgsl");
    }
    else {
        render_shader = Shader::get("data/shaders/quad_mirror.wgsl");
    }

    left_eye_texture = webgpu_context.create_texture(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_RGBA8Unorm,
        { render_width, render_height, 1 },
        WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
        1);

    right_eye_texture = webgpu_context.create_texture(
        WGPUTextureDimension_2D,
        WGPUTextureFormat_RGBA8Unorm,
        { render_width, render_height, 1 },
        WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
        1);

    u_render_texture_left_eye.data = webgpu_context.create_texture_view(left_eye_texture, WGPUTextureViewDimension_2D, WGPUTextureFormat_RGBA8Unorm);
    u_render_texture_left_eye.binding = 0;
    u_render_texture_left_eye.visibility = WGPUShaderStage_Fragment;

    u_render_texture_right_eye.data = webgpu_context.create_texture_view(right_eye_texture, WGPUTextureViewDimension_2D, WGPUTextureFormat_RGBA8Unorm);
    u_render_texture_right_eye.binding = 0;
    u_render_texture_right_eye.visibility = WGPUShaderStage_Fragment;

    // Left eye bind group
    WGPUBindGroupLayout render_bind_group_layout;
    {
        std::vector<Uniform*> uniforms = { &u_render_texture_left_eye };

        // shared with right eye
        render_bind_group_layout = webgpu_context.create_bind_group_layout(uniforms);

        render_bind_group_left_eye = webgpu_context.create_bind_group(uniforms, render_bind_group_layout);
    }

    // Right eye bind group
    {
        std::vector<Uniform*> uniforms = { &u_render_texture_right_eye };

        // render_bind_group_layout is the same as left eye
        render_bind_group_right_eye = webgpu_context.create_bind_group(uniforms, render_bind_group_layout);
    }

    render_pipeline_layout = webgpu_context.create_pipeline_layout({ render_bind_group_layout });

    // Vertex attributes
    WGPUVertexAttribute vertex_attrib_position;
    vertex_attrib_position.shaderLocation = 0;
    vertex_attrib_position.format = WGPUVertexFormat_Float32x2;
    vertex_attrib_position.offset = 0;

    WGPUVertexAttribute vertex_attrib_uv;
    vertex_attrib_uv.shaderLocation = 1;
    vertex_attrib_uv.format = WGPUVertexFormat_Float32x2;
    vertex_attrib_uv.offset = 2 * sizeof(float);

    quad_vertex_attributes = { vertex_attrib_position, vertex_attrib_uv };
    quad_vertex_layout = webgpu_context.create_vertex_buffer_layout(quad_vertex_attributes, 4 * sizeof(float), WGPUVertexStepMode_Vertex);

    quad_mesh.create_quad();

    quad_vertex_buffer = webgpu_context.create_buffer(quad_mesh.get_size() * sizeof(float), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex, quad_mesh.data());

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

    render_pipeline = webgpu_context.create_render_pipeline({ quad_vertex_layout }, color_target, render_shader->get_module(), render_pipeline_layout);
}

void RaymarchingRenderer::init_compute_raymarching_pipeline()
{
    // Load compute shader
    compute_raymarching_shader = Shader::get("data/shaders/sdf_raymarching.wgsl");

    WGPUBindGroupLayout compute_textures_bind_group_layout;
    // Texture uniforms
    {
        u_compute_texture_left_eye.data = webgpu_context.create_texture_view(left_eye_texture, WGPUTextureViewDimension_2D, WGPUTextureFormat_RGBA8Unorm);
        u_compute_texture_left_eye.binding = 0;
        u_compute_texture_left_eye.visibility = WGPUShaderStage_Compute;
        u_compute_texture_left_eye.is_storage_texture = true;
        u_compute_texture_left_eye.storage_texture_binding_layout.access = WGPUStorageTextureAccess_WriteOnly;
        u_compute_texture_left_eye.storage_texture_binding_layout.format = WGPUTextureFormat_RGBA8Unorm;
        u_compute_texture_left_eye.storage_texture_binding_layout.viewDimension = WGPUTextureViewDimension_2D;

        u_compute_texture_right_eye.data = webgpu_context.create_texture_view(right_eye_texture, WGPUTextureViewDimension_2D, WGPUTextureFormat_RGBA8Unorm);
        u_compute_texture_right_eye.binding = 1;
        u_compute_texture_right_eye.visibility = WGPUShaderStage_Compute;
        u_compute_texture_right_eye.is_storage_texture = true;
        u_compute_texture_right_eye.storage_texture_binding_layout.access = WGPUStorageTextureAccess_WriteOnly;
        u_compute_texture_right_eye.storage_texture_binding_layout.format = WGPUTextureFormat_RGBA8Unorm;
        u_compute_texture_right_eye.storage_texture_binding_layout.viewDimension = WGPUTextureViewDimension_2D;

        u_compute_texture_sdf_storage.data = webgpu_context.create_buffer(512 * 512 * 512 * sizeof(float) * 4, WGPUBufferUsage_Storage, nullptr);
        u_compute_texture_sdf_storage.binding = 2;
        u_compute_texture_sdf_storage.visibility = WGPUShaderStage_Compute;
        u_compute_texture_sdf_storage.buffer_binding_type = WGPUBufferBindingType_Storage;
        u_compute_texture_sdf_storage.buffer_size = 512 * 512 * 512 * sizeof(float) * 4;

        std::vector<Uniform*> uniforms = { &u_compute_texture_left_eye, &u_compute_texture_right_eye, &u_compute_texture_sdf_storage };

        compute_textures_bind_group_layout = webgpu_context.create_bind_group_layout(uniforms);
        compute_raymarching_textures_bind_group = webgpu_context.create_bind_group(uniforms, compute_textures_bind_group_layout);
    }

    WGPUBindGroupLayout compute_data_bind_group_layout;
    // Compute data uniforms
    {
        u_compute_buffer_data.data = webgpu_context.create_buffer(sizeof(sComputeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        u_compute_buffer_data.binding = 0;
        u_compute_buffer_data.visibility = WGPUShaderStage_Compute;
        u_compute_buffer_data.buffer_size = sizeof(sComputeData);

        std::vector<Uniform*> uniforms = { &u_compute_buffer_data };

        compute_data_bind_group_layout = webgpu_context.create_bind_group_layout(uniforms);
        compute_raymarching_data_bind_group = webgpu_context.create_bind_group(uniforms, compute_data_bind_group_layout);
    }

    WGPUPipelineLayout compute_raymarching_pipeline_layout = webgpu_context.create_pipeline_layout({ compute_textures_bind_group_layout, compute_data_bind_group_layout });

    compute_raymarching_pipeline = webgpu_context.create_compute_pipeline(compute_raymarching_shader->get_module(), compute_raymarching_pipeline_layout);
}

void RaymarchingRenderer::init_compute_merge_pipeline()
{
    // Load compute shader
    compute_merge_shader = Shader::get("data/shaders/sdf_merge.wgsl");

    WGPUBindGroupLayout compute_bind_group_layout;
    // Texture uniforms
    {
        u_compute_edits_data.data = webgpu_context.create_buffer(sizeof(edits), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        u_compute_edits_data.binding = 0;
        u_compute_edits_data.visibility = WGPUShaderStage_Compute;
        u_compute_edits_data.buffer_size = sizeof(sComputeData);

        std::vector<Uniform*> uniforms = { &u_compute_edits_data, &u_compute_texture_sdf_storage };

        compute_bind_group_layout = webgpu_context.create_bind_group_layout(uniforms);
        compute_merge_bind_group = webgpu_context.create_bind_group(uniforms, compute_bind_group_layout);
    }

    WGPUPipelineLayout compute_pipeline_layout = webgpu_context.create_pipeline_layout({ compute_bind_group_layout });

    compute_merge_pipeline = webgpu_context.create_compute_pipeline(compute_merge_shader->get_module(), compute_pipeline_layout);
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

    mirror_pipeline = webgpu_context.create_render_pipeline({ quad_vertex_layout }, color_target, mirror_shader->get_module(), render_pipeline_layout);
}

#endif