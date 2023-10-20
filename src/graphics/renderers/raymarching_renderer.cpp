#include "raymarching_renderer.h"

#include "rooms_renderer.h"
#include "framework/scene/parse_scene.h"

#include <algorithm>

RaymarchingRenderer::RaymarchingRenderer()
{
    
}

int RaymarchingRenderer::initialize(bool use_mirror_screen)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

#ifndef DISABLE_RAYMARCHER
    init_compute_raymarching_pipeline();
    //init_compute_merge_pipeline();
    init_initialize_sdf_pipeline();
    init_compute_octree_pipeline();
    init_raymarching_proxy_pipeline();

    compute_initialize_sdf();

    edits = new Edit[EDITS_MAX];

    edits[compute_merge_data.edits_to_process++] = {
        .position = { 0.0, 0.0, 0.0 },
        .primitive = SD_SPHERE,
        .color = { 1.0, 0.0, 0.0 },
        .operation = OP_SMOOTH_UNION,
        .dimensions = { 0.4, 0.4, 0.4, 0.4 },
        .rotation = { 0.f, 0.f, 0.f, 1.f },
        .parameters = { 0.0, -1.0, 0.0, 0.0 },
    };

    //edits[compute_merge_data.edits_to_process++] = {
    //    .position = { 0.1, 0.0, 0.0 },
    //    .primitive = SD_SPHERE,
    //    .color = { 0.0, 0.0, 1.0 },
    //    .operation = OP_SMOOTH_UNION,
    //    .dimensions = { 0.025, 0.025, 0.025, 0.025 },
    //    .rotation = { 0.f, 0.f, 0.f, 1.f },
    //    .parameters = { 0.0, -1.0, 0.0, 0.0 },
    //};

    //edits[compute_merge_data.edits_to_process++] = {
    //    .position = { 0.0, 0.1, 0.0 },
    //    .primitive = SD_SPHERE,
    //    .color = { 0.0, 0.0, 1.0 },
    //    .operation = OP_SMOOTH_UNION,
    //    .dimensions = { 0.025, 0.025, 0.025, 0.025 },
    //    .rotation = { 0.f, 0.f, 0.f, 1.f },
    //    .parameters = { 0.0, -1.0, 0.0, 0.0 },
    //};

#endif

    compute_raymarching_data.render_width = static_cast<float>(webgpu_context->render_width);
    compute_raymarching_data.render_height = static_cast<float>(webgpu_context->render_height);

    compute_raymarching_data.camera_far = 1000.0f;
    compute_raymarching_data.camera_near = 0.1f;

    return 0;
}

void RaymarchingRenderer::clean()
{
#ifndef DISABLE_RAYMARCHER
    // Uniforms
    compute_buffer_data_uniform.destroy();
    compute_texture_left_eye_uniform.destroy();
    compute_texture_right_eye_uniform.destroy();

    // Compute pipeline
    wgpuBindGroupRelease(compute_raymarching_textures_bind_group);
    wgpuBindGroupRelease(initialize_sdf_bind_group);
    //wgpuBindGroupRelease(compute_merge_bind_group);

    compute_texture_sdf_storage_uniform.destroy();
    //compute_texture_sdf_copy_storage_uniform.destroy();

    wgpuBindGroupRelease(compute_raymarching_data_bind_group);

    delete[] edits;
#endif
}

void RaymarchingRenderer::update(float delta_time)
{
    compute_raymarching_data.time += delta_time;

    updated_time += delta_time;
    //for (;updated_time >= 0.0166f; updated_time -= 0.0166f) {
        compute_octree();
    //}
}

void RaymarchingRenderer::render()
{

}

void RaymarchingRenderer::add_preview_edit(const Edit& edit)
{
    if (preview_edit_data.preview_edits_count >= PREVIEW_EDITS_MAX) {
        return;
    }
    preview_edit_data.preview_edits[preview_edit_data.preview_edits_count++] = edit;
}

void RaymarchingRenderer::set_sculpt_rotation(const glm::quat& rotation)
{
    compute_raymarching_data.sculpt_rotation = rotation;
    compute_merge_data.sculpt_rotation = rotation;
}

void RaymarchingRenderer::compute_initialize_sdf()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
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
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);
}

void RaymarchingRenderer::compute_octree()
{
    if (!compute_octree_evaluate_shader || !compute_octree_evaluate_shader->is_loaded()) return;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Nothing to merge if equals 0
    if (compute_merge_data.edits_to_process == 0) {
        return;
    }

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Update uniform buffer
    webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_edits_array_uniform.data), 0, edits, sizeof(Edit) * compute_merge_data.edits_to_process);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

    uint32_t default_vals[3] = { 1, 1, 1 };
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_indirect_buffer.data), 0, &default_vals, 3 * sizeof(uint32_t));

    uint32_t default_val = 0;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_current_level.data), 0, &default_val, sizeof(uint32_t));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_atomic_counter.data), 0, &default_val, sizeof(uint32_t));

    // Restore initial octant for level 0
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octant_usage_uniform[0].data), 0, &default_val, sizeof(uint32_t));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octant_usage_uniform[2].data), 0, &default_val, sizeof(uint32_t));

    int ping_pong_idx = 0;

    for (int i = 0; i <= octree_depth; ++i) {

        compute_octree_evaluate_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer.data), 0);

        compute_octree_increment_level_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        ping_pong_idx = (ping_pong_idx + 1) % 2;
    }

    compute_octree_write_to_texture_pipeline.set(compute_pass);

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_write_to_texture_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer.data), 0);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Compute Octree Command Buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);

    //compute_merge_data.edits_to_process = 0;
}

void RaymarchingRenderer::compute_merge()
{
    if (!compute_merge_shader || !compute_merge_shader->is_loaded()) return;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Nothing to merge if equals 0
    if (compute_merge_data.edits_to_process == 0) {
        return;
    }

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Compute the edit size
    glm::vec3 edit_min = { 100.0f, 100.0f, 100.0f };
    glm::vec3 edit_max = { -100.0f, -100.0f, -100.0f };
    glm::vec3 tmp_min, tmp_max;
    for (uint16_t i = 0; i < compute_merge_data.edits_to_process; i++) {
        edits[i].get_world_AABB(&tmp_min, &tmp_max, compute_merge_data.sculpt_start_position, compute_merge_data.sculpt_rotation, true);
        edit_min = glm::min(edit_min, tmp_min);
        edit_max = glm::max(edit_max, tmp_max);
    }

    edit_min = glm::clamp(edit_min, { 0.0f }, { 1.0f });
    edit_max = glm::clamp(edit_max, { 0.0f }, { 1.0f });

    // Calculate size
    glm::vec3 edit_size = edit_max - edit_min;

    // To SDF coords:
    edit_size = edit_size * static_cast<float>(SDF_RESOLUTION);
    compute_merge_data.edits_aabb_start = glm::uvec3(glm::floor(edit_min * static_cast<float>(SDF_RESOLUTION)));

    WGPUImageCopyTexture src_texture = {};
    src_texture.texture = sdf_texture.get_texture();
    src_texture.mipLevel = 0;
    src_texture.origin = { compute_merge_data.edits_aabb_start.x, compute_merge_data.edits_aabb_start.y, compute_merge_data.edits_aabb_start.z };

    WGPUImageCopyTexture dst_texture = {};
    dst_texture.texture = sdf_copy_read_texture.get_texture();
    dst_texture.mipLevel = 0;
    dst_texture.origin = { compute_merge_data.edits_aabb_start.x, compute_merge_data.edits_aabb_start.y, compute_merge_data.edits_aabb_start.z };

    uint32_t workgroupSize = 8;

    uint32_t copy_w = std::clamp(static_cast<uint32_t>(edit_size.x + workgroupSize), 0u, SDF_RESOLUTION - compute_merge_data.edits_aabb_start.x);
    uint32_t copy_h = std::clamp(static_cast<uint32_t>(edit_size.y + workgroupSize), 0u, SDF_RESOLUTION - compute_merge_data.edits_aabb_start.y);
    uint32_t copy_d = std::clamp(static_cast<uint32_t>(edit_size.z + workgroupSize), 0u, SDF_RESOLUTION - compute_merge_data.edits_aabb_start.z);

    WGPUExtent3D copy_size = { copy_w, copy_h, copy_d };
    wgpuCommandEncoderCopyTextureToTexture(command_encoder, &src_texture, &dst_texture, &copy_size);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute_raymarching pass
    compute_merge_pipeline.set(compute_pass);

    // Update uniform buffer
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(compute_edits_array_uniform.data), 0, edits, sizeof(Edit) * compute_merge_data.edits_to_process);
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_merge_bind_group, 0, nullptr);

    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth  = static_cast<uint32_t>((edit_size.x + workgroupSize - 1) / workgroupSize);
    uint32_t workgroupHeight = static_cast<uint32_t>((edit_size.y + workgroupSize - 1) / workgroupSize);
    uint32_t workgroupDepth  = static_cast<uint32_t>((edit_size.z + workgroupSize - 1) / workgroupSize);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, workgroupDepth);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Merge Command buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);

    compute_merge_data.edits_to_process = 0;
}

void RaymarchingRenderer::compute_raymarching()
{
    if (!compute_raymarching_shader || !compute_raymarching_shader->is_loaded()) return;

    // Compute thre preview edit AABB
    glm::vec3 edit_AABB_min = { 100.0f, 100.0f, 100.0f };
    glm::vec3 edit_AABB_max = { -100.0f, -100.0f, -100.0f };
    glm::vec3 tmp_min, tmp_max;
    for (uint16_t i = 0; i < preview_edit_data.preview_edits_count; i++) {
        preview_edit_data.preview_edits[i].get_world_AABB(&tmp_min, &tmp_max, glm::vec3(0.0f), compute_merge_data.sculpt_rotation, true);
        edit_AABB_min = glm::min(edit_AABB_min, tmp_min);
        edit_AABB_max = glm::max(edit_AABB_max, tmp_max);
    }

    const glm::vec3 preview_edits_AABB_half_size = (edit_AABB_max - edit_AABB_min) / 2.0f;

    preview_edit_data.aabb_center = edit_AABB_min + preview_edits_AABB_half_size;
    preview_edit_data.aabb_size = preview_edits_AABB_half_size;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Use compute_raymarching pass
    compute_raymarching_pipeline.set(compute_pass);

    // Update uniform buffer
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(compute_buffer_data_uniform.data), 0, &(compute_raymarching_data), sizeof(sComputeData));
    // Update preview edits
    // TODO: compute bounidng box of edits
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(compute_preview_edit_uniform.data), 0, &preview_edit_data, sizeof(sPreviewEditsData));

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_raymarching_textures_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_raymarching_data_bind_group, 0, nullptr);

    uint32_t workgroupSize = 16;
    // This ceils invocationCount / workgroupSize
    uint32_t workgroupWidth = (webgpu_context->render_width + workgroupSize - 1) / workgroupSize;
    uint32_t workgroupHeight = (webgpu_context->render_height + workgroupSize - 1) / workgroupSize;
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroupWidth, workgroupHeight, 1);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Raymarch Command buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);

    preview_edit_data.preview_edits_count = 0u;
}

void RaymarchingRenderer::render_raymarching_proxy(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Prepare the color attachment
    WGPURenderPassColorAttachment render_pass_color_attachment = {};
    render_pass_color_attachment.view = swapchain_view;
    render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;

    glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
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
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

    // Use render_raymarching pass
    render_proxy_geometry_pipeline.set(render_pass);

    Mesh* mesh = cube_mesh->get_mesh();

    uint8_t bind_group_index = 0;

    // Set bind groups
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, render_proxy_geometry_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, render_camera_bind_group, 0, nullptr);
    // Set vertex buffer while encoding the render pass
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, mesh->get_vertex_buffer(), 0, mesh->get_byte_size());

    // Submit indirect drawcalls
    wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(octree_proxy_indirect_buffer.data), 0u);

    wgpuRenderPassEncoderEnd(render_pass);

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Proxy geometry command buffer";

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuRenderPassEncoderRelease(render_pass);
    wgpuCommandEncoderRelease(command_encoder);
}

void RaymarchingRenderer::set_sculpt_start_position(const glm::vec3& position)
{
    compute_merge_data.sculpt_start_position = position;
    compute_raymarching_data.sculpt_start_position = position;
}

void RaymarchingRenderer::set_render_size(float width, float height)
{
    compute_raymarching_data.render_width = width;
    compute_raymarching_data.render_height = height;
}

void RaymarchingRenderer::set_left_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection)
{
    compute_raymarching_data.view_projection_left_eye = view_projection;
    compute_raymarching_data.inv_view_projection_left_eye = glm::inverse(view_projection);
    compute_raymarching_data.left_eye_pos = eye_pos;
}

void RaymarchingRenderer::set_right_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection)
{
    compute_raymarching_data.view_projection_right_eye = view_projection;
    compute_raymarching_data.inv_view_projection_right_eye = glm::inverse(view_projection);
    compute_raymarching_data.right_eye_pos = eye_pos;
}

void RaymarchingRenderer::set_near_far(float z_near, float z_far)
{
    compute_raymarching_data.camera_near = z_near;
    compute_raymarching_data.camera_far = z_far;
}

void RaymarchingRenderer::set_camera_eye(const glm::vec3& eye_pos) {
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(proxy_geometry_eye_position.data), 0,&eye_pos , sizeof(eye_pos));
}

void RaymarchingRenderer::init_compute_raymarching_pipeline()
{
    // Load compute_raymarching shader
    compute_raymarching_shader = RendererStorage::get_shader("data/shaders/sdf_raymarching.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    sdf_texture.create(
        WGPUTextureDimension_3D,
        WGPUTextureFormat_RGBA32Float,
        { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
        1, nullptr);

    //sdf_copy_read_texture.create(
    //    WGPUTextureDimension_3D,
    //    WGPUTextureFormat_RGBA32Float,
    //    { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
    //    static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopyDst),
    //    1, nullptr);

    compute_texture_sdf_storage_uniform.data = sdf_texture.get_view();
    compute_texture_sdf_storage_uniform.binding = 3;

    //compute_texture_sdf_copy_storage_uniform.data = sdf_copy_read_texture.get_view();
    //compute_texture_sdf_copy_storage_uniform.binding = 2;

    init_compute_raymarching_textures();
    

    // Compute data uniforms
    {
        compute_buffer_data_uniform.data = webgpu_context->create_buffer(sizeof(sComputeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        compute_buffer_data_uniform.binding = 0;
        compute_buffer_data_uniform.buffer_size = sizeof(sComputeData);

        compute_preview_edit_uniform.data = webgpu_context->create_buffer(sizeof(sPreviewEditsData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        compute_preview_edit_uniform.binding = 1;
        compute_preview_edit_uniform.buffer_size = sizeof(sPreviewEditsData);

        std::vector<Uniform*> uniforms = { &compute_buffer_data_uniform, &compute_preview_edit_uniform };

        compute_raymarching_data_bind_group = webgpu_context->create_bind_group(uniforms, compute_raymarching_shader, 1);
    }

    compute_raymarching_pipeline.create_compute(compute_raymarching_shader);
}

void RaymarchingRenderer::init_compute_raymarching_textures()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    if (compute_raymarching_textures_bind_group) {
        wgpuBindGroupRelease(compute_raymarching_textures_bind_group);
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

    compute_texture_left_eye_uniform.data = rooms_renderer->get_eye_texture(EYE_LEFT)->get_view();
    compute_texture_left_eye_uniform.binding = 0;

    compute_texture_right_eye_uniform.data = rooms_renderer->get_eye_texture(EYE_RIGHT)->get_view();
    compute_texture_right_eye_uniform.binding = 1;

    u_sampler.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUFilterMode_Nearest, WGPUFilterMode_Nearest, WGPUMipmapFilterMode_Nearest); // Using all default params
    u_sampler.binding = 2;

    std::vector<Uniform*> uniforms = { &compute_texture_left_eye_uniform, &compute_texture_right_eye_uniform, &compute_texture_sdf_storage_uniform, &u_sampler };

    compute_raymarching_textures_bind_group = webgpu_context->create_bind_group(uniforms, compute_raymarching_shader, 0);
}

void RaymarchingRenderer::init_initialize_sdf_pipeline()
{
    initialize_sdf_shader = RendererStorage::get_shader("data/shaders/sdf_initialization.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    std::vector<Uniform*> uniforms = { &compute_texture_sdf_storage_uniform };
    initialize_sdf_bind_group = webgpu_context->create_bind_group(uniforms, initialize_sdf_shader, 0);

    initialize_sdf_pipeline.create_compute(initialize_sdf_shader);
}

void RaymarchingRenderer::init_compute_merge_pipeline()
{
    // Load compute_raymarching shader
    compute_merge_shader = RendererStorage::get_shader("data/shaders/sdf_merge.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    //// Texture uniforms
    //{
    //    std::vector<Uniform*> uniforms = { &compute_edits_array_uniform, &compute_merge_data_uniform, &compute_texture_sdf_copy_storage_uniform, &compute_texture_sdf_storage_uniform };
    //    compute_merge_bind_group = webgpu_context->create_bind_group(uniforms, compute_merge_shader, 0);
    //}

    compute_merge_pipeline.create_compute(compute_merge_shader);
}

void RaymarchingRenderer::init_compute_octree_pipeline()
{
    // Load compute_raymarching shader
    compute_octree_evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    compute_octree_increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    compute_octree_write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // 2^3 give 8x8x8 pixel cells, and we need one iteration less, so substract 3
    octree_depth = log2(SDF_RESOLUTION) - 3;

    // Size of penultimate level
    uint32_t octants_max_size = pow(pow(2, octree_depth - 1), 3);
    uint32_t octants_max_index_buffer_size = octants_max_size * sizeof(uint32_t);
    uint32_t octants_max_position_buffer_size = octants_max_size * sizeof(glm::vec3);

    // Texture uniforms
    {
        compute_merge_data.max_octree_depth = octree_depth;

        // total size considering leaves and intermediate levels
        uint32_t total_size = (pow(8, octree_depth + 1) - 1) / 7;

        total_size *= sizeof(sOctreeNode);

        compute_edits_array_uniform.data = webgpu_context->create_buffer(sizeof(Edit) * EDITS_MAX, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        compute_edits_array_uniform.binding = 0;
        compute_edits_array_uniform.buffer_size = sizeof(Edit) * EDITS_MAX;

        compute_merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr);
        compute_merge_data_uniform.binding = 1;
        compute_merge_data_uniform.buffer_size = sizeof(sMergeData);

        octree_uniform.data = webgpu_context->create_buffer(total_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree");
        octree_uniform.binding = 2;
        octree_uniform.buffer_size = total_size;

        uint32_t default_val = 0;
        octree_current_level.data = webgpu_context->create_buffer(sizeof(uint32_t), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, &default_val, "current_level");
        octree_current_level.binding = 4;
        octree_current_level.buffer_size = sizeof(uint32_t);

        octree_atomic_counter.data = webgpu_context->create_buffer(sizeof(uint32_t), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, &default_val, "atomic_counter");
        octree_atomic_counter.binding = 5;
        octree_atomic_counter.buffer_size = sizeof(uint32_t);

        octree_proxy_instance_buffer.visibility = WGPUBufferUsage_Vertex;
        octree_proxy_instance_buffer.data = webgpu_context->create_buffer(octants_max_position_buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "proxy_boxes_position_buffer");
        octree_proxy_instance_buffer.binding = 6;
        octree_proxy_instance_buffer.buffer_size = octants_max_position_buffer_size;

        std::vector<Uniform*> uniforms = { &octree_uniform, &compute_edits_array_uniform, &compute_merge_data_uniform, &octree_atomic_counter, &octree_current_level, &octree_proxy_instance_buffer };

        compute_octree_evaluate_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 0);
    }

    {
        uint32_t default_vals[3] = { 1, 1, 1 };
        octree_indirect_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * 3, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, default_vals, "indirect_buffer");
        octree_indirect_buffer.binding = 0;
        octree_indirect_buffer.buffer_size = sizeof(uint32_t) * 3;

        EntityMesh* cube = parse_scene("data/meshes/cube/cube.obj");

        uint32_t default_indirect_buffer[4] = { cube->get_mesh()->get_vertex_count(), 0, 0 ,0};
        octree_proxy_indirect_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, default_indirect_buffer, "proxy_boxes_indirect_buffer");
        octree_proxy_indirect_buffer.binding = 2;
        octree_proxy_indirect_buffer.buffer_size = sizeof(uint32_t) * 4;

        std::vector<Uniform*> uniforms = { &octree_indirect_buffer, &octree_atomic_counter, &octree_current_level, &octree_proxy_indirect_buffer, &compute_merge_data_uniform };

        compute_octree_increment_level_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_increment_level_shader, 0);
    }

    WGPUBuffer octant_usage_buffers[2];

    uint32_t default_val = 0;

    // Ping pong buffers
    for (int i = 0; i < 2; ++i) {
        octant_usage_buffers[i] = webgpu_context->create_buffer(octants_max_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octant_usage");
        webgpu_context->update_buffer(octant_usage_buffers[i], 0, &default_val, sizeof(uint32_t));
    }

    for (int i = 0; i < 4; ++i) {
        octant_usage_uniform[i].data = octant_usage_buffers[i / 2];
        octant_usage_uniform[i].binding = i % 2;
        octant_usage_uniform[i].buffer_size = octants_max_size;
    }

    for (int i = 0; i < 2; ++i) {
        std::vector<Uniform*> uniforms = { &octant_usage_uniform[i], &octant_usage_uniform[3 - i] }; // im sorry
        compute_octant_usage_bind_groups[i] = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 1);
    }

    {
        std::vector<Uniform*> uniforms = { &compute_texture_sdf_storage_uniform, &compute_edits_array_uniform, &compute_merge_data_uniform, &octree_current_level };
        compute_octree_write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_write_to_texture_shader, 0);
    }

    compute_octree_evaluate_pipeline.create_compute(compute_octree_evaluate_shader);
    compute_octree_increment_level_pipeline.create_compute(compute_octree_increment_level_shader);
    compute_octree_write_to_texture_pipeline.create_compute(compute_octree_write_to_texture_shader);

}


void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    cube_mesh = parse_scene("data/meshes/cube/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    proxy_geometry_eye_position.data = webgpu_context->create_buffer(sizeof(glm::vec3), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "proxy geometry eye position");
    proxy_geometry_eye_position.binding = 1;
    proxy_geometry_eye_position.buffer_size = sizeof(glm::vec3);

    Uniform* camera_uniform = static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_current_camera_uniform();

    std::vector<Uniform*> uniforms = { &u_sampler, &compute_texture_sdf_storage_uniform, &octree_proxy_instance_buffer, &proxy_geometry_eye_position };

    render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);
    uniforms = { camera_uniform };
    render_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 1);

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

    render_proxy_geometry_pipeline.create_render(render_proxy_shader, color_target, true);
}
