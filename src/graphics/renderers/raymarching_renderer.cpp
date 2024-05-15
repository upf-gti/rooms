#include "raymarching_renderer.h"

#include "rooms_renderer.h"
#include "framework/scene/parse_scene.h"
#include "framework/utils/intersections.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "graphics/shader.h"
#include "graphics/renderer_storage.h"

#include "shaders/AABB_shader.wgsl.gen.h"

#include <algorithm>
#include <numeric>

#include "spdlog/spdlog.h"

#include "framework/input.h"


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
    
}

int RaymarchingRenderer::initialize(bool use_mirror_screen)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    octree_depth = static_cast<uint8_t>(OCTREE_DEPTH);

    // total size considering leaves and intermediate levels
    octree_total_size = (pow(8, octree_depth + 1) - 1) / 7;

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
    Shader::set_custom_define("STROKE_HISTORY_MAX_SIZE", STROKE_HISTORY_MAX_SIZE);
    Shader::set_custom_define("BRICK_REMOVAL_COUNT", empty_brick_and_removal_buffer_count);

    brick_world_size = (SCULPT_MAX_SIZE / octree_space_scale) * 8.0f;

#ifndef DISABLE_RAYMARCHER

    init_compute_octree_pipeline();
    init_octree_ray_intersection_pipeline();
    init_raymarching_proxy_pipeline();
    initialize_stroke();

#endif
    
    AABB_mesh = parse_mesh("data/meshes/cube/aabb_cube.obj");

    Material AABB_material = (AABB_mesh->get_surface(0)->get_material());
    //AABB_material.priority = 10;
    AABB_material.color = glm::vec4(0.8f, 0.3f, 0.9f, 1.0f);
    AABB_material.transparency_type = ALPHA_BLEND;
    AABB_material.cull_type = CULL_NONE;
    AABB_material.shader = RendererStorage::get_shader_from_source(shaders::AABB_shader::source, shaders::AABB_shader::path, AABB_material);
    //AABB_material.diffuse_texture = RendererStorage::get_texture("data/meshes/cube/cube_AABB.png");
    AABB_mesh->set_surface_material_override(AABB_mesh->get_surface(0), AABB_material);
    
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
    //updated_time += delta_time;
    //for (;updated_time >= 0.0166f; updated_time -= 0.0166f) {
    compute_octree(command_encoder);
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

void RaymarchingRenderer::add_preview_edit(const Edit& edit)
{
    if (preview_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {
        return;
    }
    preview_stroke.edits[preview_stroke.edit_count++] = edit;
}

const RayIntersectionInfo& RaymarchingRenderer::get_ray_intersection_info() const
{
    return ray_intersection_info;
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
    new_stroke.color_blending_op = params.get_color_blending_operation();
    new_stroke.parameters = params.get_parameters();
    new_stroke.material = params.get_material();
    new_stroke.edit_count = 0u;

    spdlog::info("Change stroke");
    if (current_stroke.edit_count > 0u) {
        // Add it to the history
        stroke_history.push_back(current_stroke);
    }

    current_stroke = new_stroke;
    in_frame_stroke = new_stroke;

    preview_stroke = new_stroke;
}

void RaymarchingRenderer::change_stroke(const uint32_t index_increment) {
    spdlog::info("Change stroke");
    if (current_stroke.edit_count > 0u) {
        // Add it to the history
        stroke_history.push_back(current_stroke);
    }

    current_stroke.edit_count = 0u;
}

void RaymarchingRenderer::push_edit(const Edit edit)
{
    // Check for max edits -> Prolongation of the stroke! (increment is 0)
    if (in_frame_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {
        assert(false && "This whould never happen. -Juan");
        //spdlog::info("prolongation");
    }

    in_frame_stroke.edits[in_frame_stroke.edit_count++] = edit;
}

void RaymarchingRenderer::push_stroke(const Stroke& new_stroke)
{
    to_compute_stroke_buffer.push_back(new_stroke);
    stroke_history.push_back(new_stroke);
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

    ray_info.ray_origin = ray_origin - sculpt_data.sculpt_start_position;
    ray_info.ray_origin = sculpt_data.sculpt_inv_rotation * ray_info.ray_origin;

    ray_info.ray_dir = ray_dir;
    ray_info.ray_dir = sculpt_data.sculpt_inv_rotation * ray_info.ray_dir;

    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_info_uniform.data), 0, &ray_info, sizeof(RayInfo));

    compute_octree_ray_intersection_pipeline.set(compute_pass);

    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, octree_ray_intersection_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, octree_ray_intersection_info_bind_group, 0, nullptr);

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

    // Upload preview data

    webgpu_context->update_buffer(std::get<WGPUBuffer>(preview_stroke_uniform.data), 0u, &preview_stroke, sizeof(Stroke));

    // Remove the preview tag from all the bricks
    compute_octree_brick_unmark_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_unmark_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);

    // Initializate the evaluator sequence
    compute_octree_initialization_pipeline.set(compute_pass);

    uint32_t stroke_dynamic_offset = 0;
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_initialization_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 2, preview_stroke_bind_group, 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    int ping_pong_idx = 0;

    for (int j = 0; j <= octree_depth; ++j) {

        compute_octree_evaluate_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

        compute_octree_increment_level_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        ping_pong_idx = (ping_pong_idx + 1) % 2;
    }

    preview_stroke.edit_count = 0u;
};

const Stroke* RaymarchingRenderer::compute_and_upload_context(const std::vector<Stroke>& strokes) {
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    const Stroke* strokes_to_evaluate = nullptr;

    Stroke intersection_stroke;
    uint32_t stroke_count_to_evaluate = 0u;
    uint32_t reevaluate_edit_count = 0u;

    uint32_t stroke_edit_count = 0u;
    AABB strokes_aabb;
    float max_smooth_margin = 0.0f;
    uint32_t last_history_index = stroke_history.size();
    uint32_t strokes_to_pop = 0u;

    stroke_influence_list.stroke_count = 0u;

    if (!needs_undo && !needs_redo) {

        for (uint16_t i = 0u; i < strokes.size(); i++) {
            strokes_aabb = merge_aabbs(strokes_aabb, strokes[i].get_world_AABB());
            stroke_edit_count += strokes[i].edit_count;
            max_smooth_margin = glm::max(strokes[i].parameters.w, max_smooth_margin);
        }

        strokes_to_evaluate = strokes.data();
        stroke_count_to_evaluate = strokes.size();
    } else
    if (needs_undo && (stroke_history.size() > 0u || current_stroke.edit_count > 0u)) {

        if (current_stroke.edit_count > 0u) {
            change_stroke();
        }

        uint32_t last_stroke_id = 0u;
        uint32_t stroke_count = 0u;

        uint32_t united_stroke_idx = 0;

        stroke_count_to_evaluate = 1;

        for (united_stroke_idx = stroke_history.size(); united_stroke_idx > 0; --united_stroke_idx) {
            Stroke& prev = stroke_history[united_stroke_idx - 1];

            // if stroke changes
            if (last_stroke_id != 0u && prev.stroke_id != last_stroke_id) {
                break;
            }

            strokes_aabb = merge_aabbs(strokes_aabb, prev.get_world_AABB());
            stroke_edit_count += prev.edit_count;
            max_smooth_margin = glm::max(prev.parameters.w, max_smooth_margin);
            strokes_to_pop++;

            stroke_count++;
            last_stroke_id = prev.stroke_id;
        }

        // In the case of first stroke, submit it as substraction to clear everything
        if (united_stroke_idx == 0) {
            Stroke& prev = stroke_history[0];
            prev.operation = OP_SMOOTH_SUBSTRACTION;
            strokes_to_evaluate = &prev;
            last_history_index = 0;
        }
        else {
            strokes_to_evaluate = &(stroke_history.data()[united_stroke_idx - 1]);
            last_history_index = united_stroke_idx - 1;
        }
    }

    uint32_t context_edit_count = 0u;
    strokes_aabb.half_size += max_smooth_margin;

    AABB roundded_to_grid;
    {
        glm::vec3 max_aabb = strokes_aabb.center + strokes_aabb.half_size;
        glm::vec3 min_aabb = strokes_aabb.center - strokes_aabb.half_size;

        // TODO: revisit this margin of 1 for the AABB size
        max_aabb = (glm::ceil(max_aabb / brick_world_size) + 1.0f) * brick_world_size;
        min_aabb = (glm::floor(min_aabb / brick_world_size) - 1.0f)* brick_world_size;

        roundded_to_grid.half_size = (max_aabb - min_aabb) * 0.5f;
        roundded_to_grid.center = min_aabb + roundded_to_grid.half_size;
    }

    strokes_aabb.half_size -= max_smooth_margin;

    // Get the strokes that are on the region of the undo
    for (uint32_t i = 0u; i < last_history_index; i++) {
        stroke_history[i].get_AABB_intersecting_stroke(roundded_to_grid, intersection_stroke);

        if (intersection_stroke.edit_count > 0u) {
            // reevaluate_edit_count += intersection_stroke.edit_count;
            stroke_influence_list.strokes[stroke_influence_list.stroke_count++] = intersection_stroke;
            context_edit_count += intersection_stroke.edit_count;
        }
    }

    // Include the current stroke as context
    if (current_stroke.edit_count > 0u) {
        current_stroke.get_AABB_intersecting_stroke(roundded_to_grid, intersection_stroke);

        if (intersection_stroke.edit_count > 0u) {
            // reevaluate_edit_count += intersection_stroke.edit_count;
            context_edit_count += intersection_stroke.edit_count;
            stroke_influence_list.strokes[stroke_influence_list.stroke_count++] = intersection_stroke;
        }
    }

    compute_merge_data.reevaluation_AABB_min = strokes_aabb.center - strokes_aabb.half_size;
    compute_merge_data.reevaluation_AABB_max = strokes_aabb.center + strokes_aabb.half_size;

    AABB_mesh->set_translation(roundded_to_grid.center + get_sculpt_start_position());
    AABB_mesh->scale(roundded_to_grid.half_size * 2.0f);

    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_history.data), 0, &stroke_influence_list, sizeof(sStrokeInfluence));
    // Update uniform buffer
    webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_merge_data_uniform.data), 0, &(compute_merge_data), sizeof(sMergeData));

    for (uint32_t i = 0u; i < strokes_to_pop; i++) {
        stroke_history.pop_back();
    }

    return strokes_to_evaluate;
};

void RaymarchingRenderer::evaluate_strokes(WGPUComputePassEncoder compute_pass, const Stroke* strokes_to_evaluate, bool is_undo, bool is_redo)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Stroke Evaluation");
#endif
    
    //spdlog::info("Stroke count {}, stroke edit count: {}, context size {}, total edit count: {}, avg edits per context {}", stroke_count_to_evaluate, stroke_edit_count, stroke_influence_list.stroke_count, reevaluate_edit_count, reevaluate_edit_count / (stroke_influence_list.stroke_count + 0.0001f));
    // Upload the current stroke to evaluate
    webgpu_context->update_buffer(std::get<WGPUBuffer>(compute_stroke_buffer_uniform.data), 0, strokes_to_evaluate, sizeof(Stroke));

    // New element, not undo nor redo...
    if (!(is_undo || is_redo)) {
        // Clean the redo history if something new is being evaluated
        stroke_redo_history.clear();
    }
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

        uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, preview_stroke_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        int ping_pong_idx = 0;

        for (int j = 0; j <= octree_depth; ++j) {

            compute_octree_evaluate_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_evaluate_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

            compute_octree_increment_level_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);

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
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, compute_octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

#ifndef NDEBUG
        wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

        compute_octree_increment_level_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        // Clean the texture atlas bricks dispatch
        compute_octree_brick_removal_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_indirect_brick_removal_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 8u);

    }

    compute_octree_brick_copy_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_brick_copy_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, octants_max_size / (8u * 8u * 8u), 1, 1);

    compute_octree_increment_level_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_increment_level_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
}

// TODO
void RaymarchingRenderer::compute_redo(WGPUComputePassEncoder compute_pass)
{
    //if (stroke_redo_history.size() == 0) {
    //    return;
    //}

    //std::vector<Stroke> strokes_to_evaluate;

    //uint32_t last_stroke_id = stroke_redo_history.front().stroke_id;

    //std::list<Stroke>::iterator stroke_it = stroke_redo_history.begin();
    //while (stroke_it != stroke_redo_history.end()) {
    //    // if stroke changes
    //    if (stroke_it->stroke_id != last_stroke_id) {
    //        break;
    //    }

    //    strokes_to_evaluate.push_back(*stroke_it);
    //    stroke_history.push_back(*stroke_it);

    //    stroke_it = stroke_redo_history.erase(stroke_it);
    //}

    //evaluate_strokes(compute_pass, strokes_to_evaluate.data(), false, true);
}

void RaymarchingRenderer::compute_undo(WGPUComputePassEncoder compute_pass)
{
    if (stroke_history.size() == 0 && current_stroke.edit_count == 0) {
        return;
    }

    
}

void RaymarchingRenderer::compute_octree(WGPUCommandEncoder command_encoder)
{
    if (!compute_octree_evaluate_shader || !compute_octree_evaluate_shader->is_loaded()) return;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    const bool is_going_to_evaluate = needs_undo || needs_redo || (in_frame_stroke.edit_count > 0 || to_compute_stroke_buffer.size() > 0);

    if (is_going_to_evaluate) {
        //RenderdocCapture::start_capture_frame();
    }

    // Create the octree renderpass
    WGPUComputePassDescriptor compute_pass_desc = {};
    compute_pass_desc.timestampWrites = nullptr;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    const bool needs_evaluation = in_frame_stroke.edit_count > 0 || to_compute_stroke_buffer.size() > 0;

    const Stroke* to_process = nullptr;
    if (needs_evaluation || needs_undo || needs_redo) {
        to_compute_stroke_buffer.push_back(in_frame_stroke);
        // If undo, redo or normal evaluation, we upload the context of the incomming edits
        to_process = compute_and_upload_context(to_compute_stroke_buffer);
    } else {
        // If not, just compute the context of the preview edit, and upload it
        preview_array.clear();
        preview_array.push_back(preview_stroke);
        to_process = compute_and_upload_context(preview_array);
    }
    

    if (needs_undo && to_process) {
        uint32_t set_as_preview = 0x01u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 3u, &set_as_preview, sizeof(uint32_t));
        evaluate_strokes(compute_pass, to_process, true);

    } else if (needs_evaluation) { // Merge
        spdlog::info("Evaluate stroke");
        uint32_t set_as_preview = 0u;
        spdlog::info("Stroke context count: {}, stroke edit count: {}", stroke_influence_list.stroke_count, stroke_influence_list.strokes[0].edit_count);
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 3u, &set_as_preview, sizeof(uint32_t));
        evaluate_strokes(compute_pass, to_process);

        // After evaluating the in_frame_stroke, store in the current stroke, and if full, set it to history
        for (uint32_t i = 0u; i < in_frame_stroke.edit_count; i++) {
            if (current_stroke.edit_count >= MAX_EDITS_PER_EVALUATION) {
                // Add it to the history
                stroke_history.push_back(current_stroke);
                current_stroke.edit_count = 0;
            }

            current_stroke.edits[current_stroke.edit_count++] = in_frame_stroke.edits[i];
        }

        in_frame_stroke.edit_count = 0u;

        to_compute_stroke_buffer.clear();
    }


    compute_preview_edit(compute_pass);

    //}

    // Finalize compute_raymarching pass
    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    // Copy brick counters to read buffer
    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, std::get<WGPUBuffer>(octree_brick_buffers.data), 0,
        brick_buffers_counters_read_buffer, 0, sizeof(sBrickBuffers_counters));

    //AABB_mesh->render();

    if (is_going_to_evaluate) {
        //RenderdocCapture::end_capture_frame();
    }
    needs_undo = false, needs_redo = false;
}

void RaymarchingRenderer::render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

#ifndef NDEBUG
        wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render sculpt proxy geometry");
#endif

    // Render Sculpt's Proxy geometry
    {
        render_proxy_geometry_pipeline.set(render_pass);

        // Update sculpt data
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));

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
    {
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

void RaymarchingRenderer::set_sculpt_start_position(const glm::vec3& position)
{
    sculpt_data.sculpt_start_position = position;
}

void RaymarchingRenderer::set_sculpt_rotation(const glm::quat& rotation)
{
    sculpt_data.sculpt_rotation = glm::inverse(rotation);
    sculpt_data.sculpt_inv_rotation = rotation;
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

        // Octree buffer
        std::vector<sOctreeNode> octree_default(octree_total_size+1);
        octree_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4 + octree_total_size * sizeof(sOctreeNode), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, octree_default.data(), "octree");
        octree_uniform.binding = 2;
        octree_uniform.buffer_size = sizeof(uint32_t) * 4 + octree_total_size * sizeof(sOctreeNode);

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


        // Stroke history
        size_t stroke_history_size = sizeof(sStrokeInfluence);
        octree_stroke_history.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        octree_stroke_history.binding = 6;
        octree_stroke_history.buffer_size = stroke_history_size;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_history.data), 0, &default_val, sizeof(uint32_t));

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


        std::vector<Uniform*> uniforms = { &octree_uniform, &compute_merge_data_uniform,
                                           &octree_stroke_history, &octree_brick_buffers };

        compute_octree_evaluate_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 0);
    }

    // Brick removal pass
    {
        std::vector<Uniform*> uniforms = { &octree_brick_buffers };
        compute_octree_indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_brick_removal_shader, 0);
    }

    // Octree increment iteration pass
    {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octree_uniform, &octree_brick_buffers };

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
        std::vector<Uniform*> uniforms = { &sdf_texture_uniform, &octree_uniform,
                                           &octree_stroke_history, &octree_brick_buffers, &sdf_material_texture_uniform };
        compute_octree_write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_write_to_texture_shader, 0);
    }

    uint32_t size = sizeof(Edit);
    size = sizeof(Stroke);

    {
        // Stroke buffer uniform
        compute_stroke_buffer_uniform.data = webgpu_context->create_buffer(sizeof(Stroke) * 1000, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_buffer");
        compute_stroke_buffer_uniform.binding = 0;
        compute_stroke_buffer_uniform.buffer_size = sizeof(Stroke);

        std::vector<Uniform*> uniforms = { &compute_stroke_buffer_uniform };

        compute_stroke_buffer_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 1);
    }

    // Preview data bindgroup
    {
        uint32_t struct_size = sizeof(Stroke);
        preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_bindgroup");
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

        Uniform octree_new_uniform = octree_uniform;
        octree_new_uniform.binding = 4;

        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                           &octree_new_uniform, &octree_brick_buffers };

        compute_octree_initialization_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_initialization_shader, 0);
    }

    // Clean Octree bindgroup
    {
        std::vector<Uniform*> uniforms = { &octree_uniform, &octree_indirect_buffer_struct, &octree_brick_buffers };
        compute_octree_clean_octree_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_cleaning_shader, 0);
    }


    compute_octree_evaluate_pipeline.create_compute(compute_octree_evaluate_shader);
    compute_octree_increment_level_pipeline.create_compute(compute_octree_increment_level_shader);
    compute_octree_write_to_texture_pipeline.create_compute(compute_octree_write_to_texture_shader);
    compute_octree_brick_removal_pipeline.create_compute(compute_octree_brick_removal_shader);
    compute_octree_brick_copy_pipeline.create_compute(compute_octree_brick_copy_shader);
    compute_octree_initialization_pipeline.create_compute(compute_octree_initialization_shader);
    compute_octree_cleaning_pipeline.create_compute(compute_octree_cleaning_shader);
    compute_octree_brick_unmark_pipeline.create_compute(compute_octree_brick_unmark_shader);
}


void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    cube_mesh = parse_mesh("data/meshes/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

        camera_uniform = rooms_renderer->get_current_camera_uniform();

        prev_stroke_uniform_2.data = preview_stroke_uniform.data;
        prev_stroke_uniform_2.binding = 1u;
        prev_stroke_uniform_2.buffer_size = preview_stroke_uniform.buffer_size;

        std::vector<Uniform*> uniforms = { &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &prev_stroke_uniform_2 };

        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);
        uniforms = { camera_uniform };
        render_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 1);
    }

    {
        sculpt_data_uniform.data = webgpu_context->create_buffer(sizeof(sSculptData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, &sculpt_data, "sculpt_data");
        sculpt_data_uniform.binding = 0;
        sculpt_data_uniform.buffer_size = sizeof(sSculptData);

        std::vector<Uniform*> uniforms = { &sculpt_data_uniform, &ray_intersection_info_uniform };
        sculpt_data_bind_proxy_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 2);
    }

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    PipelineDescription desc = { .cull_mode = WGPUCullMode_Back };
    render_proxy_geometry_pipeline.create_render(render_proxy_shader, color_target, desc);

    // Proxy for Preview
    render_preview_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_preview.wgsl");
    {
        std::vector<Uniform*> uniforms;

        uniforms = { camera_uniform };
        render_preview_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 0);

        uniforms = { &sculpt_data_uniform, &prev_stroke_uniform_2, &octree_brick_buffers};
        sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);
    }

    color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;
    render_preview_proxy_geometry_pipeline.create_render(render_preview_proxy_shader, color_target, desc);
}

void RaymarchingRenderer::init_octree_ray_intersection_pipeline()
{
    compute_octree_ray_intersection_shader = RendererStorage::get_shader("data/shaders/octree/octree_ray_intersection.wgsl");

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Ray Octree intersection bindgroup
    {
        linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
        linear_sampler_uniform.binding = 4;

        std::vector<Uniform*> uniforms = { &octree_uniform, &sdf_texture_uniform, &linear_sampler_uniform, &sdf_material_texture_uniform };
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

    compute_octree_ray_intersection_pipeline.create_compute(compute_octree_ray_intersection_shader);
}
