#include "raymarching_renderer.h"

#include "engine/rooms_engine.h"
#include "rooms_renderer.h"

#include "framework/scene/parse_scene.h"
#include "framework/math/intersections.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/input.h"
#include "framework/camera/camera.h"

#include "graphics/shader.h"
#include "graphics/renderer_storage.h"

#include "shaders/AABB_shader.wgsl.gen.h"

#include "tools/sculpt_editor.h"

#include <algorithm>
#include <numeric>

#include "glm/gtx/matrix_decompose.hpp"

#include "spdlog/spdlog.h"

/*
    Stroke management and lifecylce
        The begining and end of a stroke is managed by the scult_editor.

        In order to upload and manage strokes this lifecycle is managed like this
          1) The sculpt editor starts a new stroke, and sends one or more edits
          2) The incomming edits are stored in the in_frame_stroke.
          3) We obtain the context of the surrounding edits to the in_frame_stroke from the stroke_history.
          4) We send the in_frame_stroke and the context to the compute evaluation pipeline.
          5) After evaluation the in_frame_stroke, it is stored into the current_stroke, that cotais all the edits of the current stroke, from past frames.
          6) If there is enought edits in the current_stroke or the sculpt_editor signal it, we store the current_stroke to the stroke_history.
*/

RaymarchingRenderer::RaymarchingRenderer()
{
    /*AABB test = { glm::vec3(0.5, 0.0f, 1.0f), glm::vec3(0.50, 0.50f, 0.70f) };
    AABB result[8u];

    uint32_t count = stroke_manager.divide_AABB_on_max_eval_size(test, result);

    AABB test_f = { glm::vec3(0.0f), glm::vec3(0.0f) };
    for (uint32_t i = 0u; i < count; i++) {
        test_f = merge_aabbs(test_f, result[i]);
    }

    uint32_t p = 0u;*/
}

int RaymarchingRenderer::initialize(bool use_mirror_screen)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    

    // Compute constants
    

    stroke_manager.set_brick_world_size(glm::vec3(brick_world_size));

#ifndef DISABLE_RAYMARCHER
    init_compute_octree_pipeline();
    init_octree_ray_intersection_pipeline();
    init_raymarching_proxy_pipeline();
    initialize_stroke();
#endif
    
    AABB_mesh = parse_mesh("data/meshes/cube/aabb_cube.obj");

    Material* AABB_material = new Material();
    //AABB_material.priority = 10;
    AABB_material->set_color(glm::vec4(0.8f, 0.3f, 0.9f, 1.0f));
    AABB_material->set_transparency_type(ALPHA_BLEND);
    AABB_material->set_cull_type(CULL_NONE);
    AABB_material->set_type(MATERIAL_UNLIT);
    AABB_material->set_shader(RendererStorage::get_shader_from_source(shaders::AABB_shader::source, shaders::AABB_shader::path, AABB_material));
    //AABB_material.diffuse_texture = RendererStorage::get_texture("data/meshes/cube/cube_AABB.png");
    AABB_mesh->set_surface_material_override(AABB_mesh->get_surface(0), AABB_material);

    // Prepare preview stroke
    preview_stroke.edit_list.resize(PREVIEW_BASE_EDIT_LIST);
    preview_stroke.stroke.edit_count = 0u;

    preview_edit_array_length = PREVIEW_BASE_EDIT_LIST;
    
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
    wgpuBindGroupRelease(sculpt_data_bind_preview_group);
    wgpuBindGroupRelease(sculpt_data_bind_proxy_group);
    wgpuBindGroupRelease(preview_stroke_bind_group);

    delete render_proxy_shader;
    delete compute_octree_evaluate_shader;
    delete compute_octree_increment_level_shader;
    delete compute_octree_write_to_texture_shader;
    delete render_preview_proxy_shader;
#endif
}

void RaymarchingRenderer::update_sculpt(WGPUCommandEncoder command_encoder)
{
    performed_evaluation = false;

    //updated_time += delta_time;
    //for (;updated_time >= 0.0166f; updated_time -= 0.0166f) {
    RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
    compute_octree(command_encoder, engine_instance->get_current_editor_type() == EditorType::SCULPT_EDITOR);
    //}

#ifndef DISABLE_RAYMARCHER
    if (Input::is_mouse_pressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
        WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

        Camera* camera = rooms_renderer->get_camera();
        glm::vec3 ray_dir = camera->screen_to_ray(Input::get_mouse_position());

        octree_ray_intersect(camera->get_eye(), glm::normalize(ray_dir));
    }
#endif

}

AABB RaymarchingRenderer::sPreviewStroke::get_AABB() const
{
    AABB result = {};
    for (uint32_t i = 0u; i < stroke.edit_count; i++) {
        result = merge_aabbs(result, extern_get_edit_world_AABB(edit_list[i], stroke.primitive, stroke.parameters.w * 2.0f));
    }

    return result;
}

const RayIntersectionInfo& RaymarchingRenderer::get_ray_intersection_info() const
{
    return ray_intersection_info;
}

void RaymarchingRenderer::initialize_stroke()
{

}

void RaymarchingRenderer::change_stroke(const StrokeParameters& params, const uint32_t index_increment)
{
    stroke_manager.request_new_stroke(params, index_increment);
    
    preview_stroke.stroke.primitive = params.get_primitive();
    preview_stroke.stroke.operation = params.get_operation();
    preview_stroke.stroke.color_blending_op = params.get_color_blend_operation();
    preview_stroke.stroke.parameters = params.get_parameters();
    preview_stroke.stroke.material = params.get_material();
    preview_stroke.stroke.edit_count = 0u;
}

void RaymarchingRenderer::push_edit(const Edit edit)
{
    incoming_edits.push_back(edit);
}

void RaymarchingRenderer::octree_ray_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, std::function<void(glm::vec3)> callback)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
    SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
    SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

    if (!current_sculpt) {
        spdlog::warn("Can not ray intersect without current sculpt");
        return;
    }

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Convert ray, origin from world to sculpt space

    glm::quat inv_rotation = glm::inverse(current_sculpt->get_rotation());

    ray_info.ray_origin = ray_origin - current_sculpt->get_translation();
    ray_info.ray_origin = inv_rotation * ray_info.ray_origin;

    ray_info.ray_dir = ray_dir;
    ray_info.ray_dir = inv_rotation * ray_info.ray_dir;

    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_info_uniform.data), 0, &ray_info, sizeof(RayInfo));

    compute_octree_ray_intersection_pipeline.set(compute_pass);

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, octree_ray_intersection_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, octree_ray_intersection_info_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 2, sculpt_octree_bindgroup, 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);

    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, std::get<WGPUBuffer>(ray_intersection_info_uniform.data), 0, ray_intersection_info_read_buffer, 0, sizeof(RayIntersectionInfo));

    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = "Ray Intersection Command Buffer";

    // Encode and submit the GPU commands
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(command_encoder);

    webgpu_context->read_buffer(ray_intersection_info_read_buffer, sizeof(RayIntersectionInfo), &ray_intersection_info);
}

void RaymarchingRenderer::get_brick_usage(std::function<void(float, uint32_t)> callback)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    sBrickBuffers_counters brick_usage_info;
    webgpu_context->read_buffer(brick_buffers_counters_read_buffer, sizeof(sBrickBuffers_counters), &brick_usage_info);

    uint32_t brick_count = brick_usage_info.brick_instance_counter;
    float pct = brick_count / (float)max_brick_count;

    callback(pct, brick_count);
}

void RaymarchingRenderer::compute_preview_edit(WGPUComputePassEncoder compute_pass)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Preview evaluation");
#endif

    // Resize the edit buffer and rebuild the bindgroups
    if (preview_stroke.edit_list.size() > preview_edit_array_length) {
        preview_edit_array_length = preview_stroke.edit_list.size();
        uint32_t struct_size = sizeof(sToUploadStroke) + sizeof(Edit) * preview_edit_array_length;
        preview_stroke_uniform.destroy();
        preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_buffer");
        preview_stroke_uniform.binding = 0;
        preview_stroke_uniform.buffer_size = struct_size;

        prev_stroke_uniform_2.data = preview_stroke_uniform.data;
        prev_stroke_uniform_2.buffer_size = preview_stroke_uniform.buffer_size;

        std::vector<Uniform*> uniforms = { &sculpts_instance_data_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &prev_stroke_uniform_2 };
        wgpuBindGroupRelease(render_proxy_geometry_bind_group);
        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);

        uniforms = { &sculpts_instance_data_uniform, &prev_stroke_uniform_2, &octree_brick_buffers };
        wgpuBindGroupRelease(sculpt_data_bind_preview_group);
        sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);

        uniforms = { &preview_stroke_uniform };
        wgpuBindGroupRelease(preview_stroke_bind_group);
        preview_stroke_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 3);
    }

    // Upload preview data, first the stoke and tehn the edit list, since we are storing it in a vector
    webgpu_context->update_buffer(std::get<WGPUBuffer>(preview_stroke_uniform.data), 4 * sizeof(uint32_t), &(preview_stroke.stroke), sizeof(sToUploadStroke));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(preview_stroke_uniform.data), 4 * sizeof(uint32_t) + sizeof(sToUploadStroke), preview_stroke.edit_list.data(), preview_stroke.stroke.edit_count * sizeof(Edit));

    // Initializate the evaluator sequence
    compute_octree_initialization_pipeline.set(compute_pass);

    uint32_t stroke_dynamic_offset = 0;
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_initialization_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    int ping_pong_idx = 0;

    for (int j = 0; j <= octree_depth; ++j) {

        compute_octree_evaluate_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

        compute_octree_increment_level_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        ping_pong_idx = (ping_pong_idx + 1) % 2;
    }

    preview_stroke.stroke.edit_count = 0u;
#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
};

void RaymarchingRenderer::evaluate_strokes(WGPUComputePassEncoder compute_pass)
{

}

void RaymarchingRenderer::compute_delete_sculpts(WGPUComputePassEncoder compute_pass, GPUSculptData& to_delete)
{
    sculpt_delete_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, to_delete.octree_bindgroup, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, brick_buffer_bindgroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, last_octree_level_size / (8u * 8u * 8u), 1, 1);
}

void RaymarchingRenderer::compute_octree(WGPUCommandEncoder command_encoder, bool show_preview)
{
    if (!compute_octree_evaluate_shader || !compute_octree_evaluate_shader->is_loaded()) return;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Create the octree renderpass
    WGPUComputePassDescriptor compute_pass_desc = {};

    std::vector<WGPUComputePassTimestampWrites> timestampWrites(1);
    timestampWrites[0].beginningOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "pre_evaluation");
    timestampWrites[0].querySet = Renderer::instance->get_query_set();
    timestampWrites[0].endOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "evaluation");

    compute_pass_desc.timestampWrites = timestampWrites.data();

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    sToComputeStrokeData* stroke_to_compute = nullptr;

    // Sculpture deleting and cleaning
    {
        if (sculpts_to_clean.size() > 0u) {
            for (uint32_t i = 0u; i < sculpts_to_clean.size(); i++) {
                sculpts_to_clean[i].octree_uniform.destroy();
                wgpuBindGroupRelease(sculpts_to_clean[i].octree_bindgroup);
            }
            sculpts_to_clean.clear();
        }

        if (sculpts_to_delete.size() > 0u) {
#ifndef NDEBUG
            wgpuComputePassEncoderPushDebugGroup(compute_pass, "Sculpt removal");
#endif
            for (uint32_t i = 0u; i < sculpts_to_delete.size(); i++) {
                compute_delete_sculpts(compute_pass, sculpts_to_delete[i]);
                sculpts_to_clean.push_back(sculpts_to_delete[i]);
            }
            sculpts_to_delete.clear();
#ifndef NDEBUG
            wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
        }
    }

    // TODO to sculpt editor
    if (sculpts_to_process.size() > 0u) {
        // For loading a sculpt from disk
        sculpt_octree_bindgroup = sculpts_to_process.back()->get_octree_bindgroup();
        sculpt_octree_uniform = &sculpts_to_process.back()->get_octree_uniform();
        stroke_to_compute = stroke_manager.new_history_add(&sculpts_to_process.back()->get_stroke_history());
    } else if (incoming_edits.size() > 0u) {
        stroke_to_compute = stroke_manager.add(incoming_edits);
    } else if (needs_undo) {
        stroke_to_compute = stroke_manager.undo();
    } else if (needs_redo) {
        stroke_to_compute = stroke_manager.redo();
    }

    bool needs_evaluation = (stroke_to_compute != nullptr);

    if (!(show_preview && !needs_evaluation)) {
        
    }

    AABB aabb_pos;

    /*if (needs_evaluation) {

        aabb_pos = stroke_to_compute->in_frame_stroke_aabb;
        compute_merge_data.reevaluation_AABB_min = stroke_to_compute->in_frame_stroke_aabb.center - stroke_to_compute->in_frame_stroke_aabb.half_size;
        compute_merge_data.reevaluation_AABB_max = stroke_to_compute->in_frame_stroke_aabb.center + stroke_to_compute->in_frame_stroke_aabb.half_size;

        RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
        SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
        SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

        if (current_sculpt) {
            AABB_mesh->set_position(aabb_pos.center + current_sculpt->get_translation());
            AABB_mesh->set_scale(aabb_pos.half_size * 2.0f);
        }

    } else {
        AABB preview_aabb = stroke_manager.compute_grid_aligned_AABB(preview_stroke.get_AABB(), glm::vec3(brick_world_size));
        aabb_pos = preview_aabb;
        compute_merge_data.reevaluation_AABB_min = preview_aabb.center - preview_aabb.half_size;
        compute_merge_data.reevaluation_AABB_max = preview_aabb.center + preview_aabb.half_size;
    }*/

    //webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

    // Remove the preview tag from all the bricks
    {
        compute_octree_brick_unmark_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_unmark_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);
    }

    if (needs_evaluation && stroke_to_compute) {
        //upload_stroke_context_data(stroke_to_compute);
        uint32_t set_as_preview = (needs_undo) ? 0x01u : 0u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_octree_uniform->data), sizeof(uint32_t) * 2u, &set_as_preview, sizeof(uint32_t));

        spdlog::info("Evaluate stroke! id: {}, stroke context count: {}", stroke_to_compute->in_frame_stroke.stroke_id, stroke_to_compute->in_frame_influence.stroke_count);
        evaluate_strokes(compute_pass);
    }

    if (show_preview && !needs_evaluation) {
        uint32_t zero[3] = { 0u, 0u, 0u };
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_brick_buffers.data), sizeof(uint32_t), zero, sizeof(uint32_t) * 3u);
        zero[2] = 0x02u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_octree_uniform->data), 0, zero, sizeof(uint32_t) * 3u);
    }

    if (show_preview) {
        compute_preview_edit(compute_pass);
    }

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    // Copy brick counters to read buffer
    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, std::get<WGPUBuffer>(octree_brick_buffers.data), 0,
        brick_buffers_counters_read_buffer, 0, sizeof(sBrickBuffers_counters));

    //AABB_mesh->render();

    stroke_manager.update();

    needs_undo = false, needs_redo = false;
    incoming_edits.clear();

    if (sculpts_to_process.size() > 0u) {
        sculpts_to_process.pop_back();
    }
}

void RaymarchingRenderer::render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Get the number of sculpt instances and the model
    // TODO: we dont need to re-upload it each frame, inly when changed in hieraqui.. but the we need to detect changes
    {
        const uint32_t sculpt_instances_count = sculpt_instances_list.size();

        // Index buffer for the sculpt instances and their model matrices
        // TODO: create big buffer only once
        // TODO: this need to be mannaged better, not once per frame ALWAYS
        // uint32_t buffer_size = sculpt_count + sculpt_instances_list.size() * sculpt_instances_list.size();
        //uint32_t* buffer = new uint32_t[buffer_size];

        //memset(buffer, 0, sizeof(uint32_t) * buffer_size);
        sSculptInstanceData* instance_data_lists = new sSculptInstanceData[sculpt_count];
        RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
        SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
        SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

        uint32_t prev_start = 0u;
        for (uint32_t i = 0u; i < sculpt_instances_count; i++) {
            uint32_t octree_id = sculpt_instances_list[i]->get_octree_id();

            if (sculpt_instances_list[i] == current_sculpt) {
                preview_stroke.current_sculpt_idx = octree_id;
            }

            // Only set the out of focus as false if we are in teh sculpt editor
            if (current_sculpt) {
                sculpt_instances_list[i]->set_out_of_focus(sculpt_instances_list[i] != current_sculpt);
            } else {
                sculpt_instances_list[i]->set_out_of_focus(false);
            }

            instance_data_lists[octree_id].model = sculpt_instances_list[i]->get_model();
            instance_data_lists[octree_id].flags = sculpt_instances_list[i]->get_flags();

            /*uint32_t number_of_instances = buffer[octree_id] >> 20u;
            uint32_t instances_index = buffer[octree_id] & 0xFFFFF;

            buffer[(octree_id * sculpt_instances_count) + sculpt_count + number_of_instances] = i;
            buffer[octree_id] = ((number_of_instances + 1) << 20) | ((octree_id * sculpt_instances_count) + sculpt_count);*/
        }

        webgpu_context->update_buffer(std::get<WGPUBuffer>(preview_stroke_uniform.data), 0u, &(preview_stroke.current_sculpt_idx), sizeof(uint32_t));
        //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_instances_buffer_uniform.data), 0u, buffer, sizeof(uint32_t) * buffer_size);
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpts_instance_data_uniform.data), 0u, instance_data_lists, sizeof(sSculptInstanceData) * sculpt_count);

        delete[] instance_data_lists;
        //delete[] buffer;
    }

#ifndef NDEBUG
        wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render sculpt proxy geometry");
#endif

    // Render Sculpt's Proxy geometry
    {
        render_proxy_geometry_pipeline.set(render_pass);

        // Update sculpt data
        //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));

        const Surface* surface = cube_mesh->get_surface(0);

        // Set bind groups
        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_proxy_geometry_bind_group, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 1, render_camera_bind_group, 1, &camera_buffer_stride);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 2, sculpt_data_bind_proxy_group, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 3, Renderer::instance->get_lighting_bind_group(), 0, nullptr);

        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_byte_size());

        // Submit indirect drawcalls
        wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), 0u);
    }


#ifndef NDEBUG
    wgpuRenderPassEncoderPopDebugGroup(render_pass);
    wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render preview proxy geometry");
#endif
    // Render Preview proxy geometry
    if (static_cast<RoomsEngine*>(RoomsEngine::instance)->get_current_editor_type() == EditorType::SCULPT_EDITOR) {
        render_preview_proxy_geometry_pipeline.set(render_pass);

        // Update sculpt data
        //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));

        const Surface* surface = cube_mesh->get_surface(0);

        uint8_t bind_group_index = 0;

        // Set bind groups
        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_preview_camera_bind_group, 1, &camera_buffer_stride);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 1, sculpt_data_bind_preview_group, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 2, Renderer::instance->get_lighting_bind_group(), 0, nullptr);

        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_byte_size());

        // Submit indirect drawcalls
        wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 4u);
    }

#ifndef NDEBUG
    wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif
}

void RaymarchingRenderer::set_current_sculpt(SculptInstance* sculpt_instance)
{
    if (!sculpt_instance) {

    }

    stroke_manager.set_current_sculpt(sculpt_instance);

    sculpt_octree_bindgroup = sculpt_instance->get_octree_bindgroup();
    sculpt_octree_uniform = &sculpt_instance->get_octree_uniform();

    current_sculpt_id = sculpt_instance->get_octree_id();
}

void RaymarchingRenderer::init_compute_octree_pipeline()
{
    // Load shaders
    compute_octree_brick_unmark_shader = RendererStorage::get_shader("data/shaders/octree/brick_unmark.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // TO GPUGLOBALS =======================================================================================
    sdf_texture.create(
        WGPUTextureDimension_3D,
        WGPUTextureFormat_R32Float,
        { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
        1, 1, nullptr);

    sdf_texture_uniform.data = sdf_texture.get_view(WGPUTextureViewDimension_3D);
    sdf_texture_uniform.binding = 3;

    sdf_material_texture.create(
        WGPUTextureDimension_3D,
        WGPUTextureFormat_R32Uint,
        { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
        1, 1, nullptr);

    sdf_material_texture_uniform.data = sdf_material_texture.get_view(WGPUTextureViewDimension_3D);
    sdf_material_texture_uniform.binding = 8; // TODO: set as 4

    //// Size of penultimate level
    //octants_max_size = pow(floorf(SDF_RESOLUTION / 10.0f), 3.0f);

    // Uniforms & buffers for octree generation
    {
        // Edit count & other merger data
        compute_merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "merge_data");
        compute_merge_data_uniform.binding = 1;
        compute_merge_data_uniform.buffer_size = sizeof(sMergeData);

        // TO GPUGLOBALS =======================================================================================
        // Create the brick Buffers
        // An struct that contines: a empty brick counter in the atlas, the empty brick buffer, and the data off all the instances
        // TODO(Juan): remove extra memory
        uint32_t default_val = 0u;
        uint32_t struct_size =
            sizeof(uint32_t) * 4u + sizeof(uint32_t) * max_brick_count                          // Atlas empty buffer counter, padding & index buffer
            + sizeof(uint32_t) * 4u + sizeof(uint32_t) * empty_brick_and_removal_buffer_count   // brick removal counter, padding & index buffer
            + sizeof(uint32_t) * 4u + max_brick_count * sizeof(ProxyInstanceData)               // Brick counter, padding & instance buffer
            + sizeof(uint32_t) * 4u + PREVIEW_PROXY_BRICKS_COUNT * sizeof(ProxyInstanceData);   // Preview brick counter, padding & instance buffer
        std::vector<uint8_t> default_bytes(struct_size, 0);
        octree_brick_buffers.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage, default_bytes.data(), "brick_buffer_struct");
        octree_brick_buffers.binding = 5;
        octree_brick_buffers.buffer_size = struct_size;

        brick_buffers_counters_read_buffer = webgpu_context->create_buffer(sizeof(sBrickBuffers_counters), WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, nullptr, "brick counters read buffer");

        // TO GPUGLOBALS =======================================================================================
        // Empty atlas malloc data
        uint32_t* atlas_indices = new uint32_t[octants_max_size + 4u];
        atlas_indices[0] = octants_max_size;
        atlas_indices[1] = 0u;
        atlas_indices[2] = 0u;
        atlas_indices[3] = 0u;

        for (uint32_t i = 0u; i < octants_max_size; i++) {
            atlas_indices[i+4u] = octants_max_size - i - 1u;
        }

        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_brick_buffers.data), 0u, atlas_indices, sizeof(uint32_t) * (octants_max_size + 4u));
        delete[] atlas_indices;

        // Stroke Context
        

        // Buffer for indirect buffers
        


        
    }


    {
        // TO GPUGLOBALS =======================================================================================
        // Indirect buffer for octree generation compute
        octree_brick_copy_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * octants_max_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "brick_copy_buffer");
        octree_brick_copy_buffer.binding = 0;
        octree_brick_copy_buffer.buffer_size = sizeof(uint32_t) * octants_max_size;

        
    }

    // TODO HOY
    {
        // Brick unmarking bindgroup
        std::vector<Uniform*> uniforms = { &octree_brick_buffers };

        compute_octree_brick_unmark_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_unmark_shader, 0);
    }

    

    uint32_t default_val = 0;

    

    

    

    uint32_t size = sizeof(Edit);
    size = sizeof(Stroke);

    // Preview data bindgroup
    {
        uint32_t struct_size = sizeof(sToUploadStroke) + sizeof(Edit) * PREVIEW_BASE_EDIT_LIST;
        preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_buffer");
        preview_stroke_uniform.binding = 0;
        preview_stroke_uniform.buffer_size = struct_size;

        std::vector<Uniform*> uniforms = { &preview_stroke_uniform };

        preview_stroke_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 3);
    }

    

    // TODO HOY
    // Model and sculpt isntances bindgroups
    {

        uint32_t size = sizeof(uint32_t) * 4096u * 4096u; // The current max size of instances
        sculpt_instances_buffer_uniform.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 0u, "sculpt instance data");
        sculpt_instances_buffer_uniform.binding = 0u;
        sculpt_instances_buffer_uniform.buffer_size = size;

        std::vector<Uniform*> uniforms = { &sculpt_instances_buffer_uniform };
        sculpt_instances_bindgroup = webgpu_context->create_bind_group(uniforms, compute_octree_brick_copy_shader, 1u);
    }

    // Brick buffer bindgroup
    
    {
        alt_brick_uniform.binding = 0u;
        std::vector<Uniform*> uniforms = { &alt_brick_uniform };
        brick_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_delete_shader, 1u);
    }


    compute_octree_brick_unmark_pipeline.create_compute_async(compute_octree_brick_unmark_shader);
}

void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    cube_mesh = parse_mesh("data/meshes/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    // Sculpt model buffer
    {
        // TODO(Juan): make this array dinamic
        uint32_t size = sizeof(sSculptInstanceData) * 1024u; // The current max size of instances
        sculpts_instance_data_uniform.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 0u, "sculpt instance data");
        sculpts_instance_data_uniform.binding = 9u;
        sculpts_instance_data_uniform.buffer_size = size;
    }

    {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

        camera_uniform = rooms_renderer->get_current_camera_uniform();

        prev_stroke_uniform_2.data = preview_stroke_uniform.data;
        prev_stroke_uniform_2.binding = 1u;
        prev_stroke_uniform_2.buffer_size = preview_stroke_uniform.buffer_size;

        std::vector<Uniform*> uniforms = { &sculpts_instance_data_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &prev_stroke_uniform_2 };

        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);
        uniforms = { camera_uniform };
        render_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 1);
    }

    {
        //sculpt_data_uniform.data = webgpu_context->create_buffer(sizeof(sSculptData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, &sculpt_data, "sculpt_data");
        //sculpt_data_uniform.binding = 0;
        //sculpt_data_uniform.buffer_size = sizeof(sSculptData);

        std::vector<Uniform*> uniforms = { /*&sculpt_data_uniform,*/ &ray_intersection_info_uniform };
        sculpt_data_bind_proxy_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 2);
    }

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    PipelineDescription desc = { .cull_mode = WGPUCullMode_Front };
    render_proxy_geometry_pipeline.create_render_async(render_proxy_shader, color_target, desc);

    // Proxy for Preview
    render_preview_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_preview.wgsl");
    {
        std::vector<Uniform*> uniforms;

        uniforms = { camera_uniform };
        render_preview_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 0);

        uniforms = {/* &sculpt_data_uniform,*/ &prev_stroke_uniform_2, &octree_brick_buffers, &sculpts_instance_data_uniform };
        sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);
    }

    color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;
    render_preview_proxy_geometry_pipeline.create_render_async(render_preview_proxy_shader, color_target, desc);
}

void RaymarchingRenderer::init_octree_ray_intersection_pipeline()
{
    compute_octree_ray_intersection_shader = RendererStorage::get_shader("data/shaders/octree/octree_ray_intersection.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Ray Octree intersection bindgroup
    {
        linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
        linear_sampler_uniform.binding = 4;

        std::vector<Uniform*> uniforms = { &sdf_texture_uniform, &linear_sampler_uniform, &sdf_material_texture_uniform };
        octree_ray_intersection_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 0);
    }

    // Ray Octree intersection info bindgroup
    {
        ray_info_uniform.data = webgpu_context->create_buffer(sizeof(RayInfo), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "ray info");
        ray_info_uniform.binding = 0;
        ray_info_uniform.buffer_size = sizeof(RayInfo);

        ray_intersection_info_uniform.data = webgpu_context->create_buffer(sizeof(RayIntersectionInfo), WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage, nullptr, "ray intersection info");
        ray_intersection_info_uniform.binding = 3;
        ray_intersection_info_uniform.buffer_size = sizeof(RayIntersectionInfo);

        ray_intersection_info_read_buffer = webgpu_context->create_buffer(sizeof(RayIntersectionInfo), WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, nullptr, "ray intersection info read buffer");

        std::vector<Uniform*> uniforms = { &ray_info_uniform, &ray_intersection_info_uniform };
        octree_ray_intersection_info_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 1);
    }

    //{
    //    std::vector<Uniform*> uniforms = { &sculpt_data_uniform };
    //    sculpt_data_ray_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 2);
    //}

    compute_octree_ray_intersection_pipeline.create_compute_async(compute_octree_ray_intersection_shader);
}

void RaymarchingRenderer::upload_stroke_context_data(sToComputeStrokeData *stroke_to_compute)
{
   
}

void RaymarchingRenderer::create_sculpt_from_history(SculptInstance* instance, std::vector<Stroke>& stroke_history)
{
    GPUSculptData new_sculpt = create_new_sculpt();

    instance->set_sculpt_data(new_sculpt);

    sculpts_to_process.push_back(instance);
}

void RaymarchingRenderer::add_sculpt_instance(SculptInstance* instance)
{
    sculpt_instances_list.push_back(instance);
    sculpt_instance_count[instance->get_octree_id()] = 1u;
}

void RaymarchingRenderer::remove_sculpt_instance(SculptInstance* instance)
{
    auto it = std::find(sculpt_instances_list.begin(), sculpt_instances_list.end(), instance);
    if (it != sculpt_instances_list.end()) {
        sculpt_instances_list.erase(it);
        //sculpt_count--;
    }

    if (sculpt_instance_count[instance->get_octree_id()] == 1u) {
        sculpts_to_delete.push_back({
                instance->get_octree_id(),
                instance->get_octree_uniform(),
                instance->get_octree_bindgroup()
            });
        sculpt_instance_count[instance->get_octree_id()] = 0u;
    } else {
        sculpt_instance_count[instance->get_octree_id()]--;
    }
}
