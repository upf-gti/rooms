#include "raymarching_renderer.h"

#include "rooms_renderer.h"
#include "framework/scene/parse_scene.h"
#include "framework/intersections.h"

#include <algorithm>
#include <numeric>

#include "spdlog/spdlog.h"

RaymarchingRenderer::RaymarchingRenderer()
{
    
}

int RaymarchingRenderer::initialize(bool use_mirror_screen)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    octree_depth = static_cast<uint8_t>(6);

    // total size considering leaves and intermediate levels
    octree_total_size = (pow(8, octree_depth + 1) - 1) / 7;

    Shader::set_custom_define("OCTREE_DEPTH", octree_depth);
    Shader::set_custom_define("OCTREE_TOTAL_SIZE", octree_total_size);

#ifndef DISABLE_RAYMARCHER

    init_compute_octree_pipeline();
    init_raymarching_proxy_pipeline();
    initialize_stroke();

    //for (uint32_t i = 0; i < 10; i++) {
    //    push_edit({ glm::vec3(glm::vec3(0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1))),
    //                glm::vec3(0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1)),
    //                glm::vec4(0.1f, 0.1f, 0.1f, 0.1f),f
    //                glm::quat(1.0, 0.0, 0.0, 0.0)
    //    });

    //    change_stroke(SD_SPHERE, OP_UNION, { 0.f, -1.f, 0.f, 0.f });
    //}

    //edits[compute_merge_data.edits_to_process++] = 

   /* edits[compute_merge_data.edits_to_process++] = {
        .position = { 0.0f, 0.0f, 0.0f },
        .primitive = SD_SPHERE,
        .color = { 0.0, 1.0, 0.0 },
        .operation = OP_UNION,
        .dimensions = { 0.05f, 0.05f, 0.05f, 0.05f },
        .rotation = { 0.f, 0.f, 0.f, 1.f },
        .parameters = { 0.0, -1.0, 0.0, 0.0 },
    };*/

    //edits[compute_merge_data.edits_to_process++] = {
    //    .position = { -0.1, 0.0, 0.0 },
    //    .primitive = SD_SPHERE,
    //    .color = { 0.0, 1.0, 0.0 },
    //    .operation = OP_UNION,
    //    .dimensions = { 0.1f, 0.1f, 0.1f, 0.1f },
    //    .rotation = { 0.f, 0.f, 0.f, 1.f },
    //    .parameters = { 0.0, -1.0, 0.0, 0.0 },
    //};

    //edits[compute_merge_data.edits_to_process++] = {
    //    .position = { 0.2, 0.2, 0.2 },
    //    .primitive = SD_SPHERE,
    //    .color = { 1.0, 1.0, 1.0 },
    //    .operation = OP_SMOOTH_UNION,
    //    .dimensions = { 0.16f, 0.16f, 0.16f, 0.16f },
    //    .rotation = { 0.f, 0.f, 0.f, 1.f },
    //    .parameters = { 0.0, -1.0, 0.0, 0.0 },
    //};
#endif

    return 0;
}

void RaymarchingRenderer::clean()
{
#ifndef DISABLE_RAYMARCHER
    wgpuBindGroupRelease(render_proxy_geometry_bind_group);
    wgpuBindGroupRelease(compute_octree_evaluate_bind_group);
    wgpuBindGroupRelease(compute_octree_increment_level_bind_group);
    wgpuBindGroupRelease(compute_octree_write_to_texture_bind_group);
    wgpuBindGroupRelease(compute_octant_usage_bind_groups[0]);
    wgpuBindGroupRelease(compute_octant_usage_bind_groups[1]);
    wgpuBindGroupRelease(render_camera_bind_group);
    wgpuBindGroupRelease(sculpt_data_bind_group);

    delete render_proxy_shader;
    delete compute_octree_evaluate_shader;
    delete compute_octree_increment_level_shader;
    delete compute_octree_write_to_texture_shader;
#endif
}

void RaymarchingRenderer::update(float delta_time)
{
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
    //if (preview_edit_data.preview_edits_count >= PREVIEW_EDITS_MAX) {
    //    return;
    //}
    //preview_edit_data.preview_edits[preview_edit_data.preview_edits_count++] = edit;
}

void RaymarchingRenderer::initialize_stroke()
{
    in_frame_stroke.stroke_id = 0u;
}

void RaymarchingRenderer::change_stroke(const StrokeParameters& params, const uint32_t index_increment)
{
    Stroke new_stroke = {};

    new_stroke.stroke_id = current_stroke.stroke_id + index_increment;
    new_stroke.primitive = params.get_primitive();
    new_stroke.operation = params.get_operation();
    new_stroke.parameters = params.get_parameters();
    new_stroke.color = params.get_color();
    new_stroke.material = params.get_material();
    new_stroke.edit_count = 0u;

    // Only store the strokes that actually changes the sculpt
    //if (in_frame_stroke.edit_count > 0u) {
    //    // Add it to the history
    //    stroke_history.push_back(in_frame_stroke);
    //    AABB new_aabb;
    //    in_frame_stroke.get_world_AABB(&new_aabb.min, &new_aabb.max, compute_merge_data.sculpt_start_position, compute_merge_data.sculpt_rotation);
    //    stroke_history_AABB.push_back(new_aabb);
    //}

    if (current_stroke.edit_count > 0u) {
        // Add it to the history
        stroke_history.push_back(current_stroke);
    }

    current_stroke = new_stroke;
    in_frame_stroke = new_stroke;
}

void RaymarchingRenderer::push_edit(const Edit edit) {

    // Check for max edits -> Prolongation of the stroke! (increment is 0)
    if (in_frame_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {
        to_compute_stroke_buffer.push_back(in_frame_stroke);
        in_frame_stroke.edit_count = 0;
        //spdlog::info("prolongation");
    }

    in_frame_stroke.edits[in_frame_stroke.edit_count++] = edit;

    if (current_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {

        //spdlog::info("add to history");

        // Add it to the history
        stroke_history.push_back(current_stroke);
        current_stroke.edit_count = 0;
    }

    current_stroke.edits[current_stroke.edit_count++] = edit;
}

void RaymarchingRenderer::compute_preview_edit() {
    // Upload the preview stroke
    // Evaluate (mark & add proxy preview)
    // Proxy preview render
};

void RaymarchingRenderer::evaluate_strokes(const std::vector<Stroke> strokes, bool is_undo, bool is_redo)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Can be done once
    {
        if (!strokes.empty()) {
            webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_stroke_buffer_uniform.data), 0, strokes.data(), sizeof(Stroke) * strokes.size());
        }
        else {
            Stroke stroke;
            webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_stroke_buffer_uniform.data), 0, &stroke, sizeof(Stroke));
        }

        // Update uniform buffer
        webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

        uint32_t default_value = 0u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_proxy_indirect_buffer.data), sizeof(uint32_t), &default_value, sizeof(uint32_t));

        uint32_t reevaluate_aabb = is_undo ? CLEAN_BEFORE_EVAL : 0;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_state.data), sizeof(uint32_t) * 3, &reevaluate_aabb, sizeof(uint32_t));
    }

    // New element, not undo nor redo...
    if (!(is_undo || is_redo)) {
        // Clean the redo history if something new is being evaluated
        stroke_redo_history.clear();
    }

    // Must be updated per stroke, also make sure the area is reevaluated on undo
    uint16_t eval_steps = (is_undo && strokes.empty()) ? 1 : strokes.size();
    for (uint16_t i = 0; i < eval_steps; ++i)
    {
        compute_octree_initialization_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_initialization_bind_group, 0, nullptr);

        uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        int ping_pong_idx = 0;

        for (int j = 0; j <= octree_depth; ++j) {

            compute_octree_evaluate_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer.data), 0);

            compute_octree_increment_level_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            ping_pong_idx = (ping_pong_idx + 1) % 2;
        }

        // Write to texture dispatch
        compute_octree_write_to_texture_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_write_to_texture_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer.data), 0);

        // Clean the texture atlas bricks dispatch
        compute_octree_brick_removal_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_indirect_brick_removal_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_brick_removal_buffer.data), 0u);

    }

    compute_octree_brick_copy_pipeline.set(compute_pass);

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_copy_bind_group, 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);

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

}

void RaymarchingRenderer::redo()
{
    if (stroke_redo_history.size() == 0) {
        return;
    }

    std::vector<Stroke> strokes_to_evaluate;

    uint32_t last_stroke_id = stroke_redo_history.front().stroke_id;

    std::list<Stroke>::iterator stroke_it = stroke_redo_history.begin();
    while (stroke_it != stroke_redo_history.end()) {
        // if stroke changes
        if (stroke_it->stroke_id != last_stroke_id) {
            break;
        }

        strokes_to_evaluate.push_back(*stroke_it);
        stroke_history.push_back(*stroke_it);

        stroke_it = stroke_redo_history.erase(stroke_it);
    }

    //AABB new_aabb = front.get_world_AABB(compute_merge_data.sculpt_start_position, compute_merge_data.sculpt_rotation);
    //stroke_history_AABB.push_back(new_aabb);

    RenderdocCapture::start_capture_frame();

    evaluate_strokes(strokes_to_evaluate, false, true);

    RenderdocCapture::end_capture_frame();
}

void RaymarchingRenderer::undo()
{
    if (stroke_history.size() == 0 && current_stroke.edit_count == 0) {
        return;
    }

    uint32_t last_stroke_id = stroke_history.back().stroke_id;

    int united_stroke_idx;
    for (united_stroke_idx = stroke_history.size() - 1; united_stroke_idx >= 0; --united_stroke_idx) {
        // if stroke changes
        if (stroke_history[united_stroke_idx].stroke_id != last_stroke_id) {
            break;
        }
    }

    if (united_stroke_idx == -1) {
        WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

        stroke_redo_history.insert(stroke_redo_history.begin(), stroke_history.begin() + (united_stroke_idx + 1), stroke_history.end());

        stroke_history.clear();

        RenderdocCapture::start_capture_frame();
        // Initialize a command encoder
        WGPUCommandEncoderDescriptor encoder_desc = {};
        WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

        // Create compute_raymarching pass
        WGPUComputePassDescriptor compute_pass_desc = {};
        compute_pass_desc.timestampWrites = nullptr;
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

        compute_octree_cleaning_pipeline.set(compute_pass);
        uint32_t last_layer_size = pow(2, 3 * octree_depth);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_clean_octree_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, last_layer_size / (8u * 8u * 8u), 1,1);

        // Clean the texture atlas bricks dispatch
        compute_octree_brick_removal_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_indirect_brick_removal_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_brick_removal_buffer.data), 0u);

        compute_octree_brick_copy_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_copy_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);

        // Finalize compute_raymarching pass
        wgpuComputePassEncoderEnd(compute_pass);

        WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
        cmd_buff_descriptor.nextInChain = NULL;
        cmd_buff_descriptor.label = "Undo empty Octree Command Buffer";

        // Encode and submit the GPU commands
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
        wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(compute_pass);
        wgpuCommandEncoderRelease(command_encoder);
        RenderdocCapture::end_capture_frame();
    }
    else {

        AABB deleted_strokes_aabb;
        for (int i = united_stroke_idx + 1; i < stroke_history.size(); ++i) {
            deleted_strokes_aabb = merge_aabbs(deleted_strokes_aabb, stroke_history[i].get_world_AABB());
        }

        // get strokes with same id and add into the redo history
        stroke_redo_history.insert(stroke_redo_history.begin(), stroke_history.begin() + (united_stroke_idx + 1), stroke_history.end());

        stroke_history.erase(stroke_history.begin() + (united_stroke_idx + 1), stroke_history.end());

        std::vector<Stroke> strokes_to_recompute;

        // Get the strokes that are on the region of the undo
        for (uint32_t i = 0u; i < stroke_history.size(); i++) {
            AABB stroke_aabb = stroke_history[i].get_world_AABB();
            if (intersection::AABB_AABB_min_max(deleted_strokes_aabb, stroke_aabb)) {
                strokes_to_recompute.push_back(stroke_history[i]);
            }
        }

        compute_merge_data.reevaluation_AABB_min = deleted_strokes_aabb.center - deleted_strokes_aabb.half_size;
        compute_merge_data.reevaluation_AABB_max = deleted_strokes_aabb.center + deleted_strokes_aabb.half_size;

        RenderdocCapture::start_capture_frame();

        evaluate_strokes(strokes_to_recompute, true);

        RenderdocCapture::end_capture_frame();
    }
}

void RaymarchingRenderer::compute_octree()
{
    if (!compute_octree_evaluate_shader || !compute_octree_evaluate_shader->is_loaded()) return;

    // Nothing to merge if equals 0
    if (in_frame_stroke.edit_count == 0 && to_compute_stroke_buffer.size() == 0) {
        return;
    }

    RenderdocCapture::start_capture_frame();

    //spdlog::debug(in_frame_stroke.edits[0].to_string());

    to_compute_stroke_buffer.push_back(in_frame_stroke);

    evaluate_strokes(to_compute_stroke_buffer);

    in_frame_stroke.edit_count = 0u;

    to_compute_stroke_buffer.clear();

    RenderdocCapture::end_capture_frame();
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
    render_pass_color_attachment.loadOp = WGPULoadOp_Load;
    render_pass_color_attachment.storeOp = WGPUStoreOp_Store;

    glm::vec4 clear_color = RoomsRenderer::instance->get_clear_color();
    render_pass_color_attachment.clearValue = WGPUColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

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
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_descr);

    // Use render_raymarching pass
    render_proxy_geometry_pipeline.set(render_pass);

    // Update sculpt data
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));

    Mesh* mesh = cube_mesh->get_surface(0).mesh;

    uint8_t bind_group_index = 0;

    // Set bind groups
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, render_proxy_geometry_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, render_camera_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, sculpt_data_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, bind_group_index++, Renderer::instance->get_ibl_bind_group(), 0, nullptr);

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
    sculpt_data.sculpt_start_position = position;
}

void RaymarchingRenderer::set_sculpt_rotation(const glm::quat& rotation)
{
    sculpt_data.sculpt_rotation = glm::inverse(rotation);
    sculpt_data.sculpt_inv_rotation = rotation;
    compute_merge_data.sculpt_rotation = rotation;
}

void RaymarchingRenderer::set_camera_eye(const glm::vec3& eye_pos) {
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(proxy_geometry_eye_position.data), 0,&eye_pos , sizeof(eye_pos));
}

void RaymarchingRenderer::init_compute_octree_pipeline()
{
    // Load compute_raymarching shader
    compute_octree_evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    compute_octree_increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    compute_octree_write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");
    compute_octree_brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    compute_octree_brick_copy_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy.wgsl");
    compute_octree_initialization_shader = RendererStorage::get_shader("data/shaders/octree/initialization.wgsl");
    compute_octree_cleaning_shader = RendererStorage::get_shader("data/shaders/octree/clean_octree.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    sdf_texture.create(
        WGPUTextureDimension_3D,
        WGPUTextureFormat_R32Float,
        { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
        1, nullptr);

    sdf_texture_uniform.data = sdf_texture.get_view();
    sdf_texture_uniform.binding = 3;

    sdf_material_texture.create(
        WGPUTextureDimension_3D,
        WGPUTextureFormat_R32Uint,
        { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
        1, nullptr);

    sdf_material_texture_uniform.data = sdf_material_texture.get_view();
    sdf_material_texture_uniform.binding = 8; // TODO: set as 4

    // Size of penultimate level
    octants_max_size = pow(floorf(SDF_RESOLUTION / 10.0f), 3.0f);

    // Uniforms & buffers for octree generation
    {
        compute_merge_data.max_octree_depth = octree_depth;

        // Edit count & other merger data
        compute_merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "merge_data");
        compute_merge_data_uniform.binding = 1;
        compute_merge_data_uniform.buffer_size = sizeof(sMergeData);

        // Octree buffer
        std::vector<sOctreeNode> octree_default(octree_total_size);
        octree_uniform.data = webgpu_context->create_buffer(octree_total_size * sizeof(sOctreeNode), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, octree_default.data(), "octree");
        octree_uniform.binding = 2;
        octree_uniform.buffer_size = octree_total_size * sizeof(sOctreeNode);

        // Counters for octree merge
        uint32_t default_vals_zero[4] = { 0, 0, 0, 0 };
        octree_state.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, default_vals_zero, "octree_state");
        octree_state.binding = 4;
        octree_state.buffer_size = sizeof(uint32_t) * 4;

        // Proxy geometry instance data
        // An struct that contines: a empty brick counter in the atlas, the empty brick buffer, and the data off all the instances
        // TODO clean this section
        uint32_t default_val = 0u;
        uint32_t struct_size = sizeof(uint32_t) + sizeof(uint32_t) * octants_max_size + octants_max_size * sizeof(ProxyInstanceData);
        std::vector<uint8_t> default_bytes(struct_size, 0);
        octree_proxy_instance_buffer.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, default_bytes.data(), "proxy_boxes_position_buffer");
        octree_proxy_instance_buffer.binding = 5;
        octree_proxy_instance_buffer.buffer_size = struct_size;

        // Empty atlas malloc data
        uint32_t* atlas_indices = new uint32_t[octants_max_size + 1u];
        atlas_indices[0] = octants_max_size;

        for (uint32_t i = 0u; i < octants_max_size; i++) {
            atlas_indices[i+1u] = octants_max_size - i - 1u;
        }

        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_proxy_instance_buffer.data), 0, atlas_indices, sizeof(uint32_t) * (octants_max_size + 1));
        delete[] atlas_indices;

        // Edit culling lists per octree node buffer and culling count per octree node layer
        uint32_t edit_culling_data_size = octree_total_size * MAX_EDITS_PER_EVALUATION * sizeof(uint32_t) / 4 + octree_total_size * sizeof(uint32_t);
        octree_edit_culling_data.data = webgpu_context->create_buffer(edit_culling_data_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "edit_culling_data");
        octree_edit_culling_data.binding = 6;
        octree_edit_culling_data.buffer_size = edit_culling_data_size;

        // Buffer for brick removal & indirect buffers
        // 3 uints for the indirect buffer data + 1 padding +  and then the brick size
        uint32_t buffer_removal_buffer_size = sizeof(uint32_t) + octants_max_size * sizeof(uint32_t);
        octree_indirect_brick_removal_buffer.data = webgpu_context->create_buffer(buffer_removal_buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect | WGPUBufferUsage_Storage, nullptr, "indirect_brick_removal");
        octree_indirect_brick_removal_buffer.binding = 8;
        octree_indirect_brick_removal_buffer.buffer_size = buffer_removal_buffer_size;

        uint32_t default_removal_indirect[4] = {0, 1, 1, 0};
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_indirect_brick_removal_buffer.data), 0, default_removal_indirect, sizeof(uint32_t) * 4u);


        std::vector<Uniform*> uniforms = { &octree_uniform, &compute_merge_data_uniform,
                                           &octree_state, &octree_proxy_instance_buffer, &octree_edit_culling_data, &octree_indirect_brick_removal_buffer };

        compute_octree_evaluate_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 0);
    }

    {
        std::vector<Uniform*> uniforms = { &octree_proxy_instance_buffer, &octree_indirect_brick_removal_buffer };
        compute_octree_indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_removal_shader, 0);
    }

    {
        // Indirect buffer for octree generation compute
        uint32_t default_vals[3] = { 1, 1, 1 };
        octree_indirect_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * 3, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, default_vals, "indirect_buffer");
        octree_indirect_buffer.binding = 0;
        octree_indirect_buffer.buffer_size = sizeof(uint32_t) * 3;

        EntityMesh* cube = parse_mesh("data/meshes/cube/cube.obj");

        // Indirect rendering of proxy geometry config buffer
        uint32_t default_indirect_buffer[4] = { cube->get_surface(0).mesh->get_vertex_count(), 0, 0 ,0};
        octree_proxy_indirect_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, default_indirect_buffer, "proxy_boxes_indirect_buffer");
        octree_proxy_indirect_buffer.binding = 2;
        octree_proxy_indirect_buffer.buffer_size = sizeof(uint32_t) * 4;

        std::vector<Uniform*> uniforms = { &octree_indirect_buffer, &octree_state };

        compute_octree_increment_level_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_increment_level_shader, 0);
    }

    {
        // Indirect buffer for octree generation compute
        octree_brick_copy_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * octants_max_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "brick_copy_buffer");
        octree_brick_copy_buffer.binding = 0;
        octree_brick_copy_buffer.buffer_size = sizeof(uint32_t) * octants_max_size;

        std::vector<Uniform*> uniforms = { &octree_brick_copy_buffer, &octree_proxy_instance_buffer, &octree_proxy_indirect_buffer };

        compute_octree_brick_copy_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_copy_shader, 0);
    }

    WGPUBuffer octant_usage_buffers[2];

    uint32_t default_val = 0;

    // Ping pong buffers for read & write octants for the octree compute
    for (int i = 0; i < 2; ++i) {
        octant_usage_buffers[i] = webgpu_context->create_buffer(octants_max_size * sizeof(uint32_t), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octant_usage");
        webgpu_context->update_buffer(octant_usage_buffers[i], 0, &default_val, sizeof(uint32_t));
    }

    for (int i = 0; i < 4; ++i) {
        octant_usage_uniform[i].data = octant_usage_buffers[i / 2];
        octant_usage_uniform[i].binding = i % 2;
        octant_usage_uniform[i].buffer_size = octants_max_size * sizeof(uint32_t);
    }

    for (int i = 0; i < 2; ++i) {
        std::vector<Uniform*> uniforms = { &octant_usage_uniform[i], &octant_usage_uniform[3 - i] }; // im sorry
        compute_octant_usage_bind_groups[i] = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 2);
    }

    {
        std::vector<Uniform*> uniforms = { &sdf_texture_uniform, &octree_uniform,
                                           &octree_state, &octree_edit_culling_data, &octree_proxy_instance_buffer, &sdf_material_texture_uniform };
        compute_octree_write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_write_to_texture_shader, 0);
    }

    {
        // Stroke buffer uniform
        compute_stroke_buffer_uniform.data = webgpu_context->create_buffer(sizeof(Stroke) * 1000, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_buffer");
        compute_stroke_buffer_uniform.binding = 0;
        compute_stroke_buffer_uniform.buffer_size = sizeof(Stroke);

        std::vector<Uniform*> uniforms = { &compute_stroke_buffer_uniform };

        compute_stroke_buffer_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 1);
    }

    {
        octant_usage_initialization_uniform[0].data = octant_usage_uniform[0].data;
        octant_usage_initialization_uniform[0].binding = 1;
        octant_usage_initialization_uniform[0].buffer_size = octant_usage_uniform[0].buffer_size;

        octant_usage_initialization_uniform[1].data = octant_usage_uniform[2].data;
        octant_usage_initialization_uniform[1].binding = 2;
        octant_usage_initialization_uniform[1].buffer_size = octant_usage_uniform[1].buffer_size;


        std::vector<Uniform*> uniforms = { &octree_indirect_buffer, &octree_state, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                           &octree_edit_culling_data, &octree_indirect_brick_removal_buffer };

        compute_octree_initialization_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_initialization_shader, 0);
    }

    // Clean Octree bindgroup
    {
        Uniform proxy_indirect = octree_proxy_indirect_buffer;
        proxy_indirect.binding = 6;

        std::vector<Uniform*> uniforms = { &octree_uniform, &octree_indirect_brick_removal_buffer, &proxy_indirect };
        compute_octree_clean_octree_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_cleaning_shader, 0);
    }

    compute_octree_evaluate_pipeline.create_compute(compute_octree_evaluate_shader);
    compute_octree_increment_level_pipeline.create_compute(compute_octree_increment_level_shader);
    compute_octree_write_to_texture_pipeline.create_compute(compute_octree_write_to_texture_shader);
    compute_octree_brick_removal_pipeline.create_compute(compute_octree_brick_removal_shader);
    compute_octree_brick_copy_pipeline.create_compute(compute_octree_brick_copy_shader);
    compute_octree_initialization_pipeline.create_compute(compute_octree_initialization_shader);
    compute_octree_cleaning_pipeline.create_compute(compute_octree_cleaning_shader);
}


void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    cube_mesh = parse_mesh("data/meshes/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    {
        proxy_geometry_eye_position.data = webgpu_context->create_buffer(sizeof(glm::vec3), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "proxy geometry eye position");
        proxy_geometry_eye_position.binding = 1;
        proxy_geometry_eye_position.buffer_size = sizeof(glm::vec3);

        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

        Uniform* camera_uniform = rooms_renderer->get_current_camera_uniform();

        linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
        linear_sampler_uniform.binding = 2;

        std::vector<Uniform*> uniforms = { &linear_sampler_uniform, &sdf_texture_uniform, &octree_proxy_instance_buffer, &proxy_geometry_eye_position, &octree_brick_copy_buffer, &sdf_material_texture_uniform };

        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);
        uniforms = { camera_uniform };
        render_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 1);
    }

    {
        sculpt_data_uniform.data = webgpu_context->create_buffer(sizeof(sSculptData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, &sculpt_data, "sculpt_data");
        sculpt_data_uniform.binding = 0;
        sculpt_data_uniform.buffer_size = sizeof(sSculptData);

        std::vector<Uniform*> uniforms = { &sculpt_data_uniform };
        sculpt_data_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 2);
    }

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    PipelineDescription desc = { .cull_mode = WGPUCullMode_Back };
    render_proxy_geometry_pipeline.create_render(render_proxy_shader, color_target, desc);
}
