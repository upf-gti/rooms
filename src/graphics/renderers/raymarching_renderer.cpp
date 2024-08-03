#include "raymarching_renderer.h"

#include "engine/rooms_engine.h"
#include "rooms_renderer.h"
#include "framework/scene/parse_scene.h"
#include "framework/math/intersections.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/input.h"
#include "graphics/shader.h"
#include "graphics/renderer_storage.h"

#include "shaders/AABB_shader.wgsl.gen.h"

#include "tools/sculpt/sculpt_editor.h"

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

    octree_depth = static_cast<uint8_t>(OCTREE_DEPTH);

    // total size considering leaves and intermediate levels
    octree_total_size = (pow(8, octree_depth + 1) - 1) / 7;

    last_octree_level_size = octree_total_size - (pow(8, octree_depth) - 1) / 7;

    uint32_t s = sizeof(ProxyInstanceData);

    // Compute constants
    uint32_t brick_count_in_axis = static_cast<uint32_t>(SDF_RESOLUTION / BRICK_SIZE);
    max_brick_count = brick_count_in_axis * brick_count_in_axis * brick_count_in_axis;

    empty_brick_and_removal_buffer_count = max_brick_count + (max_brick_count % 4);
    float octree_space_scale = powf(2.0, octree_depth + 3);

    // Scale the size of a brick
    Shader::set_custom_define("WORLD_SPACE_SCALE", octree_space_scale); // Worldspace scale is 1/octree_max_width
    Shader::set_custom_define("OCTREE_DEPTH", octree_depth);
    Shader::set_custom_define("OCTREE_TOTAL_SIZE", octree_total_size);
    Shader::set_custom_define("PREVIEW_PROXY_BRICKS_COUNT", PREVIEW_PROXY_BRICKS_COUNT);
    Shader::set_custom_define("BRICK_REMOVAL_COUNT", empty_brick_and_removal_buffer_count);
    Shader::set_custom_define("MAX_SUBDIVISION_SIZE", last_octree_level_size);
    Shader::set_custom_define("MAX_STROKE_INFLUENCE_COUNT", max_stroke_influence_count);

    brick_world_size = (SCULPT_MAX_SIZE / octree_space_scale) * 8.0f;

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
    AABB_material->color = glm::vec4(0.8f, 0.3f, 0.9f, 1.0f);
    AABB_material->transparency_type = ALPHA_BLEND;
    AABB_material->cull_type = CULL_NONE;
    AABB_material->type = MATERIAL_UNLIT;
    AABB_material->shader = RendererStorage::get_shader_from_source(shaders::AABB_shader::source, shaders::AABB_shader::path, AABB_material);
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

    // Initialize a command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    // Create compute_raymarching pass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // Convert ray, origin from world to sculpt space

    RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
    SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
    SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

    if (!current_sculpt) {
        spdlog::warn("Can not ray intersect without current sculpt");
        return;
    }

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

    // Print output
    struct UserData {
        WGPUBuffer read_buffer;
        RayIntersectionInfo* info;
        bool finished = false;
    } user_data;

    user_data.read_buffer = ray_intersection_info_read_buffer;
    user_data.info = &ray_intersection_info;

    wgpuBufferMapAsync(ray_intersection_info_read_buffer, WGPUMapMode_Read, 0, sizeof(RayIntersectionInfo), [](WGPUBufferMapAsyncStatus status, void* user_data_ptr) {

        UserData* user_data = static_cast<UserData*>(user_data_ptr);

        if (status == WGPUBufferMapAsyncStatus_Success) {
            *user_data->info = *(const RayIntersectionInfo*)wgpuBufferGetConstMappedRange(user_data->read_buffer, 0, sizeof(RayIntersectionInfo));
            wgpuBufferUnmap(user_data->read_buffer);
        }

        user_data->finished = true;

    }, &user_data);

    while (!user_data.finished) {
        // Checks for ongoing asynchronous operations and call their callbacks if needed
        webgpu_context->process_events();

        if (user_data.info->intersected > 0 && callback) {
            callback(user_data.info->intersection_position);
        }
    }
}

void RaymarchingRenderer::get_brick_usage(std::function<void(float, uint32_t)> callback)
{
    struct UserData {
        WGPUBuffer read_buffer;
        sBrickBuffers_counters info;
        bool finished = false;
    } user_data;

    user_data.read_buffer = brick_buffers_counters_read_buffer;

    wgpuBufferMapAsync(user_data.read_buffer, WGPUMapMode_Read, 0, sizeof(sBrickBuffers_counters), [](WGPUBufferMapAsyncStatus status, void* user_data_ptr) {

        UserData* user_data = static_cast<UserData*>(user_data_ptr);

        if (status == WGPUBufferMapAsyncStatus_Success) {
            user_data->info = *(const sBrickBuffers_counters*)wgpuBufferGetConstMappedRange(user_data->read_buffer, 0, sizeof(sBrickBuffers_counters));
            wgpuBufferUnmap(user_data->read_buffer);
        }

        user_data->finished = true;

        }, &user_data);

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    while (!user_data.finished) {
        // Checks for ongoing asynchronous operations and call their callbacks if needed
        webgpu_context->process_events();

        uint32_t brick_count = user_data.info.brick_instance_counter;
        float pct = brick_count / (float)max_brick_count;

        callback(pct, brick_count);
    }
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

        std::vector<Uniform*> uniforms = { &sculpt_model_buffer_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &prev_stroke_uniform_2 };
        wgpuBindGroupRelease(render_proxy_geometry_bind_group);
        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);

        uniforms = { &sculpt_model_buffer_uniform, &prev_stroke_uniform_2, &octree_brick_buffers };
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
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    performed_evaluation = true;

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Stroke Evaluation");
#endif

    //TODO: the evaluator only handle ONE stroke, the other should be added to the history/context
    // First pass: evaluate the incomming strokes
    // Must be updated per stroke, also make sure the area is reevaluated on undo
    uint16_t eval_steps = 1;//(is_undo && strokes.empty()) ? 1 : strokes.size();
    for (uint16_t i = 0; i < eval_steps; ++i)
    {
#ifndef NDEBUG
        wgpuComputePassEncoderPushDebugGroup(compute_pass, "Octree evaluation");
#endif
        compute_octree_initialization_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_initialization_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);

        //uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview_stroke_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        int ping_pong_idx = 0;

        for (int j = 0; j <= octree_depth; ++j) {

            compute_octree_evaluate_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

            compute_octree_increment_level_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);

            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            ping_pong_idx = (ping_pong_idx + 1) % 2;
        }

#ifndef NDEBUG
        wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

#ifndef NDEBUG
        wgpuComputePassEncoderPushDebugGroup(compute_pass, "Write to texture");
#endif
        // Write to texture dispatch
        compute_octree_write_to_texture_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_write_to_texture_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

#ifndef NDEBUG
        wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

        compute_octree_increment_level_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_octree_bindgroup, 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        // Clean the texture atlas bricks dispatch
        compute_octree_brick_removal_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_indirect_brick_removal_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 8u);

    }

#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
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
        // rset the brick instance counter
        uint32_t zero = 0u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_brick_buffers.data), sizeof(uint32_t), &(zero), sizeof(uint32_t));
    }

    AABB aabb_pos;

    if (needs_evaluation) {

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
    }

    webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

    // Remove the preview tag from all the bricks
    {
        compute_octree_brick_unmark_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_unmark_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);
    }

    if (needs_evaluation && stroke_to_compute) {
        upload_stroke_context_data(stroke_to_compute);
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

    compute_octree_brick_copy_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_copy_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_instances_bindgroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);

    compute_octree_increment_level_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

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
        // uint32_t buffer_size = sculpt_count + sculpt_instances_list.size() * sculpt_instances_list.size();
        //uint32_t* buffer = new uint32_t[buffer_size];

        //memset(buffer, 0, sizeof(uint32_t) * buffer_size);
        glm::mat4* matrices_list = new glm::mat4[sculpt_count];
        RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
        SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
        SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

        uint32_t prev_start = 0u;
        for (uint32_t i = 0u; i < sculpt_instances_count; i++) {
            uint32_t octree_id = sculpt_instances_list[i]->get_octree_id();

            matrices_list[octree_id] = sculpt_instances_list[i]->get_model();

            if (sculpt_instances_list[i] == current_sculpt) {
                preview_stroke.current_sculpt_idx = octree_id;
            }

            /*uint32_t number_of_instances = buffer[octree_id] >> 20u;
            uint32_t instances_index = buffer[octree_id] & 0xFFFFF;

            buffer[(octree_id * sculpt_instances_count) + sculpt_count + number_of_instances] = i;
            buffer[octree_id] = ((number_of_instances + 1) << 20) | ((octree_id * sculpt_instances_count) + sculpt_count);*/
        }

        webgpu_context->update_buffer(std::get<WGPUBuffer>(preview_stroke_uniform.data), 0u, &(preview_stroke.current_sculpt_idx), sizeof(uint32_t));
        //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_instances_buffer_uniform.data), 0u, buffer, sizeof(uint32_t) * buffer_size);
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_model_buffer_uniform.data), 0u, matrices_list, sizeof(glm::mat4) * sculpt_count);

        delete[] matrices_list;
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
    stroke_manager.set_current_sculpt(sculpt_instance);

    sculpt_octree_bindgroup = sculpt_instance->get_octree_bindgroup();
    sculpt_octree_uniform = &sculpt_instance->get_octree_uniform();
}

void RaymarchingRenderer::init_compute_octree_pipeline()
{
    std::vector<std::string> define_specializations;

    // Enable SSAA for SDF write to texture
    if (SSAA_SDF_WRITE_TO_TEXTURE) {
        define_specializations.push_back("SSAA_SDF_WRITE_TO_TEXTURE");
    }

    // Load compute_raymarching shader
    compute_octree_evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    compute_octree_increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    compute_octree_write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl", define_specializations);
    compute_octree_brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    compute_octree_brick_copy_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy.wgsl");
    compute_octree_initialization_shader = RendererStorage::get_shader("data/shaders/octree/initialization.wgsl");
    compute_octree_cleaning_shader = RendererStorage::get_shader("data/shaders/octree/clean_octree.wgsl");
    compute_octree_brick_unmark_shader = RendererStorage::get_shader("data/shaders/octree/brick_unmark.wgsl");
    sculpt_delete_shader = RendererStorage::get_shader("data/shaders/octree/sculpture_delete.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

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

    // Size of penultimate level
    octants_max_size = pow(floorf(SDF_RESOLUTION / 10.0f), 3.0f);

    // Uniforms & buffers for octree generation
    {
        // Edit count & other merger data
        compute_merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "merge_data");
        compute_merge_data_uniform.binding = 1;
        compute_merge_data_uniform.buffer_size = sizeof(sMergeData);

        // Counters for octree merge
        uint32_t default_vals_zero[4] = { 0, 0, 0, 0 };
        octree_state.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, default_vals_zero, "octree_state");
        octree_state.binding = 4;
        octree_state.buffer_size = sizeof(uint32_t) * 4;

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
        stroke_context_size = STROKE_CONTEXT_INTIAL_SIZE;
        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * STROKE_CONTEXT_INTIAL_SIZE;
        octree_stroke_context.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        octree_stroke_context.binding = 6;
        octree_stroke_context.buffer_size = stroke_history_size;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), 0, &default_val, sizeof(uint32_t));

        // Culling list
        uint32_t culling_size = sizeof(uint32_t) * (2u * last_octree_level_size * max_stroke_influence_count);
        stroke_culling_data.data = webgpu_context->create_buffer(culling_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_culling_data");
        stroke_culling_data.binding = 9;
        stroke_culling_data.buffer_size = culling_size;

        // Stroke Edit list
        octree_edit_list_size = EDIT_BUFFER_INITIAL_SIZE;
        size_t edit_list_size = sizeof(Edit) * octree_edit_list_size;
        octree_edit_list.data = webgpu_context->create_buffer(edit_list_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree_edit_list");
        octree_edit_list.binding = 7;
        octree_edit_list.buffer_size = edit_list_size;

        // Buffer for indirect buffers
        uint32_t buffer_size = sizeof(uint32_t) * 4u * 4u;
        octree_indirect_buffer_struct.data = webgpu_context->create_buffer(buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect | WGPUBufferUsage_Storage, nullptr, "indirect_buffers_struct");
        octree_indirect_buffer_struct.binding = 8u;
        octree_indirect_buffer_struct.buffer_size = buffer_size;

        uint32_t default_indirect_values[16u] = {
            36u, 0u, 0u, 0u, // bricks indirect call
            36u, 0u, 0u, 0u,// preview bricks indirect call
            0u, 1u, 1u, 0u, // brick removal call (1 padding)
            1u, 1u, 1u, 0u // octree subdivision
        };
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), 0, default_indirect_values, sizeof(uint32_t) * 16u);


        std::vector<Uniform*> uniforms = { &compute_merge_data_uniform, &octree_edit_list,
                                           &octree_stroke_context, &octree_brick_buffers, &stroke_culling_data };

        compute_octree_evaluate_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 0);
    }

    // Brick removal pass
    {
        std::vector<Uniform*> uniforms = { &octree_brick_buffers };
        compute_octree_indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_removal_shader, 0);
    }

    // Octree increment iteration pass
    {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octree_brick_buffers };

        compute_octree_increment_level_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_increment_level_shader, 0);
    }

    {
        // Indirect buffer for octree generation compute
        octree_brick_copy_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * octants_max_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "brick_copy_buffer");
        octree_brick_copy_buffer.binding = 0;
        octree_brick_copy_buffer.buffer_size = sizeof(uint32_t) * octants_max_size;

        std::vector<Uniform*> uniforms = { &octree_brick_copy_buffer, &octree_brick_buffers };

        compute_octree_brick_copy_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_copy_shader, 0);
    }

    {
        // Brick unmarking bindgroup
        std::vector<Uniform*> uniforms = { &octree_brick_buffers };

        compute_octree_brick_unmark_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_unmark_shader, 0);
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
        octree_indirect_buffer_struct_2 = octree_indirect_buffer_struct;
        octree_indirect_buffer_struct_2.binding = 7u;
        std::vector<Uniform*> uniforms = { &sdf_texture_uniform, &octree_edit_list, &stroke_culling_data,
                                           &octree_stroke_context, &octree_brick_buffers, &sdf_material_texture_uniform };
        compute_octree_write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_write_to_texture_shader, 0);
    }

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

    // Octree initialiation bindgroup
    {
        octant_usage_initialization_uniform[0].data = octant_usage_uniform[0].data;
        octant_usage_initialization_uniform[0].binding = 1;
        octant_usage_initialization_uniform[0].buffer_size = octant_usage_uniform[0].buffer_size;

        octant_usage_initialization_uniform[1].data = octant_usage_uniform[2].data;
        octant_usage_initialization_uniform[1].binding = 2;
        octant_usage_initialization_uniform[1].buffer_size = octant_usage_uniform[1].buffer_size;


        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                            &octree_brick_buffers, &stroke_culling_data, &octree_stroke_context };

        compute_octree_initialization_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_initialization_shader, 0);
    }

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
    Uniform alt_brick_uniform = octree_brick_buffers;
    {
        alt_brick_uniform.binding = 0u;
        std::vector<Uniform*> uniforms = { &alt_brick_uniform };
        brick_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_delete_shader, 1u);
    }

    compute_octree_evaluate_pipeline.create_compute_async(compute_octree_evaluate_shader);
    compute_octree_increment_level_pipeline.create_compute_async(compute_octree_increment_level_shader);
    compute_octree_write_to_texture_pipeline.create_compute_async(compute_octree_write_to_texture_shader);
    compute_octree_brick_removal_pipeline.create_compute_async(compute_octree_brick_removal_shader);
    compute_octree_brick_copy_pipeline.create_compute_async(compute_octree_brick_copy_shader);
    compute_octree_initialization_pipeline.create_compute_async(compute_octree_initialization_shader);
    compute_octree_brick_unmark_pipeline.create_compute_async(compute_octree_brick_unmark_shader);
    sculpt_delete_pipeline.create_compute_async(sculpt_delete_shader);
}

void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    cube_mesh = parse_mesh("data/meshes/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    // Sculpt model buffer
    {
        uint32_t size = sizeof(glm::mat4) * 4096u; // The current max size of instances
        sculpt_model_buffer_uniform.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 0u, "sculpt instance data");
        sculpt_model_buffer_uniform.binding = 9u;
        sculpt_model_buffer_uniform.buffer_size = size;
    }

    {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

        camera_uniform = rooms_renderer->get_current_camera_uniform();

        prev_stroke_uniform_2.data = preview_stroke_uniform.data;
        prev_stroke_uniform_2.binding = 1u;
        prev_stroke_uniform_2.buffer_size = preview_stroke_uniform.buffer_size;

        std::vector<Uniform*> uniforms = { &sculpt_model_buffer_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &prev_stroke_uniform_2 };

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

        uniforms = {/* &sculpt_data_uniform,*/ &prev_stroke_uniform_2, &octree_brick_buffers, &sculpt_model_buffer_uniform };
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
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool recreated_edit_buffer = false, recreated_stroke_context_buffer = false;

    // First check if the GPU buffers need to be resized
    if (stroke_manager.edit_list_count > octree_edit_list_size) {
        spdlog::info("Resized GPU edit buffer from {} to {}", octree_edit_list_size, stroke_manager.edit_list.size());

        octree_edit_list_size = stroke_manager.edit_list.size();
        octree_edit_list.destroy();

        size_t edit_list_size = sizeof(Edit) * octree_edit_list_size;
        octree_edit_list.data = webgpu_context->create_buffer(edit_list_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "edit_list");
        octree_edit_list.binding = 7;
        octree_edit_list.buffer_size = edit_list_size;

        recreated_edit_buffer = true;
    }

    if (stroke_to_compute->in_frame_influence.strokes.size() > stroke_context_size) {
        spdlog::info("Resized GPU stroke context buffer from {} to {}", stroke_context_size, stroke_to_compute->in_frame_influence.strokes.size());

        stroke_context_size = stroke_to_compute->in_frame_influence.strokes.size();

        octree_stroke_context.destroy();

        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * stroke_context_size;
        octree_stroke_context.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        octree_stroke_context.binding = 6;
        octree_stroke_context.buffer_size = stroke_history_size;

        recreated_stroke_context_buffer = true;
    }

    // If one of the buffers was recreated/resized, then recreate the necessary bindgroups
    if (recreated_stroke_context_buffer || recreated_edit_buffer) {
        std::vector<Uniform*> uniforms = { &sdf_texture_uniform, &octree_edit_list, &stroke_culling_data,
                                           &octree_stroke_context, &octree_brick_buffers, &sdf_material_texture_uniform };

        wgpuBindGroupRelease(compute_octree_write_to_texture_bind_group);
        compute_octree_write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_write_to_texture_shader, 0);


        uniforms = { &compute_merge_data_uniform, &octree_edit_list,
                     &octree_stroke_context, &octree_brick_buffers, &stroke_culling_data };

        wgpuBindGroupRelease(compute_octree_evaluate_bind_group);
        compute_octree_evaluate_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 0);
    }

    if (recreated_stroke_context_buffer) {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                        &octree_brick_buffers, &stroke_culling_data, &octree_stroke_context };

        compute_octree_initialization_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_initialization_shader, 0);
    }

    // Upload the data to the GPU
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_edit_list.data), 0, stroke_manager.edit_list.data(), sizeof(Edit) * stroke_manager.edit_list_count);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), 0, &stroke_to_compute->in_frame_influence, sizeof(uint32_t) * 4 * 4);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), sizeof(uint32_t) * 4 * 4, stroke_to_compute->in_frame_influence.strokes.data(), stroke_to_compute->in_frame_influence.stroke_count * sizeof(sToUploadStroke));

    //spdlog::info("min aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_min.x, stroke_to_compute->in_frame_influence.eval_aabb_min.y, stroke_to_compute->in_frame_influence.eval_aabb_min.z);
    //spdlog::info("max aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_max.x, stroke_to_compute->in_frame_influence.eval_aabb_max.y, stroke_to_compute->in_frame_influence.eval_aabb_max.z);
}

GPUSculptData RaymarchingRenderer::create_new_sculpt()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    Uniform octree_uniform;
    std::vector<sOctreeNode> octree_default(octree_total_size + 1);
    octree_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4 + octree_total_size * sizeof(sOctreeNode), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree");
    octree_uniform.binding = 0;
    octree_uniform.buffer_size = sizeof(uint32_t) * 4u + octree_total_size * sizeof(sOctreeNode);
    // Set the id of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 3, &sculpt_count, sizeof(uint32_t));
    // Set default values of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 4, octree_default.data(), sizeof(sOctreeNode) * octree_total_size);

    std::vector<Uniform*> uniforms = { &octree_uniform };
    WGPUBindGroup octree_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 1);

    sculpt_instance_count.push_back(1u);

    return { sculpt_count++, octree_uniform, octree_buffer_bindgroup };
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
