#include "sculpt_manager.h"

#include "rooms_includes.h"

#include "engine/rooms_engine.h"

#include "graphics/shader.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "framework/resources/sculpt.h"
#include "framework/nodes/sculpt_node.h"

#include <spdlog/spdlog.h>

void get_mapped_result_buffer(WGPUBufferMapAsyncStatus status, void* user_payload);

void SculptManager::init()
{
    init_shaders();
    init_uniforms();
    init_pipelines_and_bindgroups();
}

void SculptManager::clean()
{
#ifndef DISABLE_RAYMARCHER
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

    wgpuBufferRelease(read_results.gpu_results_read_buffer);
    wgpuBindGroupRelease(evaluate_bind_group);
    wgpuBindGroupRelease(increment_level_bind_group);
    wgpuBindGroupRelease(write_to_texture_bind_group);
    wgpuBindGroupRelease(octant_usage_ping_pong_bind_groups[0]);
    wgpuBindGroupRelease(octant_usage_ping_pong_bind_groups[1]);
    wgpuBindGroupRelease(gpu_results_bindgroup);
    wgpuBindGroupRelease(ray_sculpt_info_bind_group);
    wgpuBindGroupRelease(ray_intersection_info_bind_group);
    wgpuBindGroupRelease(sdf_atlases_sampler_bindgroup);
    wgpuBindGroupRelease(indirect_brick_removal_bind_group);
    wgpuBindGroupRelease(sculpt_delete_bindgroup);
    wgpuBindGroupRelease(brick_unmark_bind_group);
    wgpuBindGroupRelease(preview_stroke_bind_group);

    gpu_results_uniform.destroy();
    ray_info_uniform.destroy();
    ray_sculpt_instances_uniform.destroy();
    ray_intersection_info_uniform.destroy();
    octree_edit_list.destroy();
    stroke_context_list.destroy();
    stroke_culling_buffer.destroy();
    octant_usage_ping_pong_uniforms[0].destroy();
    octant_usage_ping_pong_uniforms[1].destroy();
    octant_usage_initialization_uniform[0].destroy();

    delete evaluate_shader;
    delete increment_level_shader;
    delete write_to_texture_shader;
    delete brick_removal_shader;
    delete brick_copy_shader;
    delete brick_unmark_shader;
    delete sculpt_delete_shader;
    delete ray_intersection_shader;
    delete evaluation_initialization_shader;
    delete ray_intersection_result_and_clean_shader;
#endif
}

void SculptManager::update(WGPUCommandEncoder command_encoder)
{
    // New render pass for the interseccions
    if (intersections_to_compute > 0u) {
        WGPUComputePassDescriptor compute_pass_desc = {};

        std::vector<WGPUComputePassTimestampWrites> timestampWrites(1);
        timestampWrites[0].beginningOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "pre_evaluation_or_what");
        timestampWrites[0].querySet = Renderer::instance->get_query_set();
        timestampWrites[0].endOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "intersection");

        compute_pass_desc.timestampWrites = timestampWrites.data();

        WGPUComputePassEncoder intersection_compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

        evaluate_closest_ray_intersection(intersection_compute_pass);

        wgpuComputePassEncoderEnd(intersection_compute_pass);
        wgpuComputePassEncoderRelease(intersection_compute_pass);
    }

    // Create the octree renderpass
    WGPUComputePassDescriptor compute_pass_desc = {};

    std::vector<WGPUComputePassTimestampWrites> timestampWrites(1);
    timestampWrites[0].beginningOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "pre_evaluation");
    timestampWrites[0].querySet = Renderer::instance->get_query_set();
    timestampWrites[0].endOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "evaluation");

    compute_pass_desc.timestampWrites = timestampWrites.data();

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    if (!evaluations_to_process.empty()) {
        sEvaluateRequest& evaluate_request = evaluations_to_process.back();

        evaluate(compute_pass, evaluate_request);

        evaluations_to_process.pop_back();
    }

    if (previus_dispatch_had_preview) {
        clean_previous_preview(compute_pass);
    }

    if (preview.needs_computing) {
        upload_preview_strokes();
        evaluate_preview(compute_pass);
    }

     // Sculpture deleting and cleaning
    {
        if (sculpts_to_clean.size() > 0u) {
            for (uint32_t i = 0u; i < sculpts_to_clean.size(); i++) {
                sculpts_to_clean[i]->get_octree_uniform().destroy();
                wgpuBindGroupRelease(sculpts_to_clean[i]->get_octree_bindgroup());
                delete sculpts_to_clean[i];
            }
            sculpts_to_clean.clear();
        }

        if (sculpts_to_delete.size() > 0u) {
#ifndef NDEBUG
            wgpuComputePassEncoderPushDebugGroup(compute_pass, "Sculpt removal");
#endif
            for (uint32_t i = 0u; i < sculpts_to_delete.size(); i++) {
                delete_sculpt(compute_pass, sculpts_to_delete[i]);
                sculpts_to_clean.push_back(sculpts_to_delete[i]);
            }
            sculpts_to_delete.clear();
#ifndef NDEBUG
            wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
        }
    }

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    if (intersections_to_compute > 0u) {
        wgpuCommandEncoderCopyBufferToBuffer(command_encoder, std::get<WGPUBuffer>(gpu_results_uniform.data), 0u, read_results.gpu_results_read_buffer, 0u, sizeof(sGPU_SculptResults));
        intersections_to_compute = 0u;
    }
}

void SculptManager::update_sculpt(Sculpt* sculpt, const sStrokeInfluence& strokes_to_process, const uint32_t edit_count, const std::vector<Edit>& edits_to_process)
{
    evaluations_to_process.push_back({ sculpt, strokes_to_process, edit_count, edits_to_process });
}

void SculptManager::set_preview_stroke(Sculpt* sculpt, const uint32_t in_gpu_model_idx, sGPUStroke preview_stroke, const std::vector<Edit>& preview_edits)
{
    preview.to_upload_stroke = preview_stroke;
    preview.to_upload_edit_list = &preview_edits;
    preview.to_upload_stroke.edit_count = preview_edits.size();
    preview.sculpt = sculpt;
    preview.sculpt_model_idx = in_gpu_model_idx;

    preview.needs_computing = true;

    static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->set_preview_render(true);
}

void SculptManager::set_ray_to_test(const glm::vec3& ray_origin, const glm::vec3& ray_dir, SculptNode *node_to_test)
{
    if (intersections_to_compute > 0u) {
        spdlog::error("Only one ray test per frame!");
        assert(0u);
    }

    ray_to_upload.ray_origin = ray_origin;
    ray_to_upload.ray_direction = ray_dir;
    intersections_to_compute = 1u;
    intersection_node_to_test = node_to_test;
}

Sculpt* SculptManager::create_sculpt()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    Uniform octree_uniform;
    std::vector<sOctreeNode> octree_default(sdf_globals.octree_total_size + 1);
    octree_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4 + sdf_globals.octree_total_size * sizeof(sOctreeNode), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree");
    octree_uniform.binding = 0;
    octree_uniform.buffer_size = sizeof(uint32_t) * 4u + sdf_globals.octree_total_size * sizeof(sOctreeNode);
    // Set the id of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 3, &sculpt_count, sizeof(uint32_t));
    // Set default values of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 4, octree_default.data(), sizeof(sOctreeNode) * sdf_globals.octree_total_size);

    Uniform brick_index_buffer;
    const size_t brick_index_size = sizeof(uint32_t) * sdf_globals.octree_last_level_size;
    brick_index_buffer.data = webgpu_context->create_buffer(brick_index_size, WGPUBufferUsage_Storage, nullptr, "brick index buffer");
    brick_index_buffer.binding = 1u;
    brick_index_buffer.buffer_size = brick_index_size;

    Uniform indirect_buffer;
    const size_t indirect_buffer_size = sizeof(uint32_t) * 6;
    uint32_t values[6] = {36u, 0u, 0u, 0u, 0u, 0u};
    indirect_buffer.data = webgpu_context->create_buffer(indirect_buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, values, "brick indirect buffer");
    indirect_buffer.binding = 2u;
    indirect_buffer.buffer_size = indirect_buffer_size;

    std::vector<Uniform*> uniforms = { &octree_uniform, &brick_index_buffer, &indirect_buffer };
    WGPUBindGroup evaluate_sculpt_bindgroup = webgpu_context->create_bind_group(uniforms, brick_copy_shader, 0u, "Write read octree bindgroup");

    uniforms = { &octree_uniform };
    WGPUBindGroup octree_bindgroup = webgpu_context->create_bind_group(uniforms, evaluate_shader, 1u, "Octree bindgroup");

    uniforms = { &brick_index_buffer, &indirect_buffer };
    WGPUBindGroup readonly_sculpt_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl"), 2u, "Read only octree bindgroup");

    uniforms = { &octree_uniform, &indirect_buffer };
    WGPUBindGroup oct_indi_bindgroup = webgpu_context->create_bind_group(uniforms, evaluation_initialization_shader, 1u, "Read only octree bindgroup");

    return new Sculpt(sculpt_count++, octree_uniform, indirect_buffer, brick_index_buffer, octree_bindgroup, evaluate_sculpt_bindgroup, oct_indi_bindgroup, readonly_sculpt_buffer_bindgroup);
}

Sculpt* SculptManager::create_sculpt_from_history(const std::vector<Stroke>& stroke_history)
{
    Sculpt* new_sculpt = create_sculpt();
    new_sculpt->set_stroke_history(stroke_history);

    sEvaluateRequest sculpt_creation;
    sculpt_creation.sculpt = new_sculpt;
    sculpt_creation.strokes_to_process.strokes.resize(stroke_history.size());

    AABB eval_aabb;

    for (const Stroke& curr_stroke : stroke_history) {
        const AABB& curr_stroke_aabb = curr_stroke.get_world_AABB();
        eval_aabb = merge_aabbs(eval_aabb, curr_stroke_aabb);

        sGPUStroke to_upload_curr;
        to_upload_curr.edit_count = curr_stroke.edit_count;
        to_upload_curr.edit_list_index = sculpt_creation.edit_count;
        to_upload_curr.material = curr_stroke.material;
        to_upload_curr.stroke_id = curr_stroke.stroke_id;
        to_upload_curr.primitive = curr_stroke.primitive;
        to_upload_curr.operation = curr_stroke.operation;
        to_upload_curr.color_blending_op = curr_stroke.color_blending_op;
        to_upload_curr.aabb_min = curr_stroke_aabb.center - curr_stroke_aabb.half_size;
        to_upload_curr.aabb_max = curr_stroke_aabb.center + curr_stroke_aabb.half_size;
        to_upload_curr.parameters = curr_stroke.parameters;

        sculpt_creation.strokes_to_process.strokes[sculpt_creation.strokes_to_process.stroke_count++] = to_upload_curr;

        sculpt_creation.edit_to_process.resize(sculpt_creation.edit_to_process.size() + curr_stroke.edit_count);

        for (uint32_t i = 0u; i < curr_stroke.edit_count; i++) {
            sculpt_creation.edit_to_process[sculpt_creation.edit_count++] = curr_stroke.edits[i];
        }
    }

    sculpt_creation.strokes_to_process.eval_aabb_min = eval_aabb.center - eval_aabb.half_size;
    sculpt_creation.strokes_to_process.eval_aabb_max = eval_aabb.center + eval_aabb.half_size;

    evaluations_to_process.push_back(sculpt_creation);

    return new_sculpt;
}

void SculptManager::delete_sculpt(WGPUComputePassEncoder compute_pass, Sculpt* to_delete)
{
    if (!sculpt_delete_pipeline.is_loaded()) {
        return;
    }
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Delete Sculpt");
#endif
    sculpt_delete_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, to_delete->get_octree_bindgroup(), 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_delete_bindgroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octree_last_level_size / (8u * 8u * 8u), 1, 1);
#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
}

void SculptManager::evaluate(WGPUComputePassEncoder compute_pass, const sEvaluateRequest& evaluate_request)
{
    if (!evaluate_shader || !evaluate_shader->is_loaded()) return;
    if (!evaluation_initialization_pipeline.is_loaded() ||
        !evaluate_pipeline.is_loaded() ||
        !increment_level_pipeline.is_loaded() ||
        !write_to_texture_pipeline.is_loaded() ||
        !brick_copy_pipeline.is_loaded()) {
        return;
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    spdlog::info("Evaluating sculpt {}, from ({},{},{}) to ({},{},{})",
        evaluate_request.sculpt->get_sculpt_id(),
        evaluate_request.strokes_to_process.eval_aabb_min.x,
        evaluate_request.strokes_to_process.eval_aabb_min.y,
        evaluate_request.strokes_to_process.eval_aabb_min.z,
        evaluate_request.strokes_to_process.eval_aabb_max.x,
        evaluate_request.strokes_to_process.eval_aabb_max.y,
        evaluate_request.strokes_to_process.eval_aabb_max.z);

    // Prepare for evaluation
    // Reset the brick instance counter
    uint32_t zero = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.brick_buffers.data), sizeof(uint32_t), &(zero), sizeof(uint32_t));

    webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), 0, &evaluate_request.strokes_to_process, sizeof(uint32_t) * 4 * 4);

    // TODO: debug render eval AABB
    uint32_t set_as_preview = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluate_request.sculpt->get_octree_uniform().data), sizeof(uint32_t) * 2u, &set_as_preview, sizeof(uint32_t));
    // Upload strokes
    upload_strokes_and_edits(evaluate_request.strokes_to_process.stroke_count, evaluate_request.strokes_to_process.strokes, evaluate_request.edit_count, evaluate_request.edit_to_process);

    // Compute dispatches
    {
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
            evaluation_initialization_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluation_initialization_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_indirect_bindgroup(), 0u, nullptr);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

            //uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview_stroke_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            int ping_pong_idx = 0;
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);


            for (int j = 0; j <= sdf_globals.octree_depth; ++j) {

                evaluate_pipeline.set(compute_pass);

                wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluate_bind_group, 0, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_ping_pong_bind_groups[ping_pong_idx], 0, nullptr);

                wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 12u);

                increment_level_pipeline.set(compute_pass);

                wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, gpu_results_bindgroup, 0u, nullptr);
                //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0u, nullptr);

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
            write_to_texture_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, write_to_texture_bind_group, 0, nullptr);
            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0u, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_ping_pong_bind_groups[ping_pong_idx], 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 12u);

#ifndef NDEBUG
            wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

            increment_level_pipeline.set(compute_pass);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, gpu_results_bindgroup, 0u, nullptr);
            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0u, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            // Clean the texture atlas bricks dispatch
            brick_removal_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, indirect_brick_removal_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 8u);

        }

#ifndef NDEBUG
        wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

        brick_copy_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, evaluate_request.sculpt->get_sculpt_bindgroup(), 0u, nullptr);

        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octree_last_level_size / (8u * 8u * 8u), 1, 1);

        increment_level_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    }

    performed_evaluation = true;
}

void SculptManager::clean_previous_preview(WGPUComputePassEncoder compute_pass)
{
    if (!brick_unmark_pipeline.is_loaded()) {
        return;
    }
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    // Call the brick unmark
#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Brick unmark");
#endif
    brick_unmark_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, brick_unmark_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octants_max_size / (8u * 8u * 8u), 1, 1);
#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif
    previus_dispatch_had_preview = false;
}

void SculptManager::evaluate_preview(WGPUComputePassEncoder compute_pass)
{
    if (!evaluation_initialization_pipeline.is_loaded() ||
        !evaluate_pipeline.is_loaded() ||
        !increment_level_pipeline.is_loaded()) {
        return;
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    // Set preview flag if needed
    if (!performed_evaluation) {
        uint32_t set_as_preview = 0x02u;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(preview.sculpt->get_octree_uniform().data), sizeof(uint32_t) * 2u, &set_as_preview, sizeof(uint32_t));
    }

    upload_preview_strokes();

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Preview evaluation");
#endif

    // Initializate the evaluator sequence
    evaluation_initialization_pipeline.set(compute_pass);

    uint32_t stroke_dynamic_offset = 0;
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluation_initialization_bind_group, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview.sculpt->get_octree_indirect_bindgroup(), 0, nullptr);

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    int ping_pong_idx = 0;

    for (int j = 0; j <= sdf_globals.octree_depth; ++j) {

        evaluate_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluate_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview.sculpt->get_octree_bindgroup(), 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_ping_pong_bind_groups[ping_pong_idx], 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 12u);

        increment_level_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, gpu_results_bindgroup, 0u, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

        ping_pong_idx = (ping_pong_idx + 1) % 2;
    }

#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

    preview.needs_computing = false;
    previus_dispatch_had_preview = true;
}

void SculptManager::evaluate_closest_ray_intersection(WGPUComputePassEncoder compute_pass)
{
    if (!ray_intersection_pipeline.is_loaded() ||
        !ray_intersection_result_and_clean_pipeline.is_loaded()) {
        return;
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Ray intersection");
#endif

    // TODO: add range checks for the ray, for an early out
    // TODO: if bindless, we could do a workgroup/thread per ray
 
    // Upload ray uniform
    uint32_t starting_idx = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_info_uniform.data), 0u, &ray_to_upload, sizeof(sGPU_RayData));
    if (intersection_node_to_test != nullptr) {
        starting_idx = intersection_node_to_test->get_in_frame_render_instance_idx();
    } 
    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_sculpt_instances_uniform.data), 0u, &starting_idx, sizeof(uint32_t));
    

    ray_intersection_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, ray_intersection_info_bind_group, 0u, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, ray_sculpt_info_bind_group, 0u, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 3u, sdf_atlases_sampler_bindgroup, 0u, nullptr);

    if (intersection_node_to_test == nullptr) {
        for (auto& it : rooms_renderer->get_sculpts_render_list()) {
            Sculpt* curr_sculpt = it.second->sculpt;
            const uint32_t instances_count = it.second->instance_count;

            wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, curr_sculpt->get_octree_bindgroup(), 0u, nullptr);

            for (uint32_t i = 0u; i < instances_count; i++) {
                wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);
            }
        }
    } else {
        Sculpt* curr_sculpt = intersection_node_to_test->get_sculpt_data();

        wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, curr_sculpt->get_octree_bindgroup(), 0u, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);
    }

    ray_intersection_result_and_clean_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, gpu_results_bindgroup, 0u, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

    intersection_node_to_test = nullptr;
    performed_evaluation = true;
}

void SculptManager::upload_strokes_and_edits(const uint32_t stroke_count, const std::vector<sGPUStroke>& strokes_to_compute, const uint32_t edits_count, const std::vector<Edit>& edits_to_upload)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    bool recreated_edit_buffer = false, recreated_stroke_context_buffer = false;

    // TODO: Expand buffers by chunks

    // First check if the GPU buffers need to be resized
    if (edits_to_upload.size() > octree_edit_list_size) {
        spdlog::info("Resized GPU edit buffer from {} to {}", octree_edit_list_size, edits_to_upload.size());

        octree_edit_list_size = edits_to_upload.size();
        octree_edit_list.destroy();

        size_t edit_list_size = sizeof(Edit) * octree_edit_list_size;
        octree_edit_list.data = webgpu_context->create_buffer(edit_list_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "edit_list");
        octree_edit_list.binding = 7;
        octree_edit_list.buffer_size = edit_list_size;

        recreated_edit_buffer = true;
    }

    if (strokes_to_compute.size() > stroke_context_list_size) {
        spdlog::info("Resized GPU stroke context buffer from {} to {}", stroke_context_list_size, strokes_to_compute.size());

        stroke_context_list_size = strokes_to_compute.size();

        stroke_context_list.destroy();

        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sGPUStroke) * stroke_context_list_size;
        stroke_context_list.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        stroke_context_list.binding = 6;
        stroke_context_list.buffer_size = stroke_history_size;

        recreated_stroke_context_buffer = true;
    }

    // If one of the buffers was recreated/resized, then recreate the necessary bindgroups
    if (recreated_stroke_context_buffer || recreated_edit_buffer) {
        std::vector<Uniform*> uniforms = { &sdf_globals.sdf_texture_uniform, &octree_edit_list, &stroke_culling_buffer,
                                           &stroke_context_list, &sdf_globals.brick_buffers, &sdf_globals.sdf_material_texture_uniform };

        wgpuBindGroupRelease(write_to_texture_bind_group);
        write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);


        uniforms = { &octree_edit_list,
                     &stroke_context_list, &sdf_globals.brick_buffers, &stroke_culling_buffer };

        wgpuBindGroupRelease(evaluate_bind_group);
        evaluate_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 0);
    }

    if (recreated_stroke_context_buffer) {
        std::vector<Uniform*> uniforms = { &sdf_globals.indirect_buffers, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                        &sdf_globals.brick_buffers, &stroke_culling_buffer, &stroke_context_list };

        evaluation_initialization_bind_group = webgpu_context->create_bind_group(uniforms, evaluation_initialization_shader, 0);
    }

    // TODO: This is sending all the edits & strokes from the buffer. The array is created once
    // Upload the data to the GPU
    spdlog::info("   - Context edit size: {}", edits_count);
    spdlog::info("   - Context stroke size: {}", stroke_count);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_edit_list.data), 0, edits_to_upload.data(), sizeof(Edit) * edits_count);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), sizeof(uint32_t) * 4 * 4, strokes_to_compute.data(), sizeof(sGPUStroke) * stroke_count);
}

void SculptManager::upload_preview_strokes()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    //// Resize the edit buffer and rebuild the bindgroups
    //if (to_upload_preview_edit_list.size() > preview_edit_array_length) {
    //    preview_edit_array_length = to_upload_preview_edit_list.size();
    //    uint32_t struct_size = sizeof(sGPUStroke) + sizeof(Edit) * preview_edit_array_length;
    //    sdf_globals.preview_stroke_uniform.destroy();
    //    sdf_globals.preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_buffer");
    //    sdf_globals.preview_stroke_uniform.binding = 0;
    //    sdf_globals.preview_stroke_uniform.buffer_size = struct_size;

    //    sdf_globals.preview_stroke_uniform_2.data = sdf_globals.preview_stroke_uniform.data;
    //    sdf_globals.preview_stroke_uniform_2.buffer_size = sdf_globals.preview_stroke_uniform.buffer_size;
    //    std::vector<Uniform*> uniforms = { &sculpts_instance_data_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_buffer, &sdf_material_texture_uniform, &preview_stroke_uniform_2 };
    //    wgpuBindGroupRelease(render_proxy_geometry_bind_group);
    //    render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);

    //    uniforms = { &sculpts_instance_data_uniform, &preview_stroke_uniform_2, &octree_brick_buffers };
    //    wgpuBindGroupRelease(sculpt_data_bind_preview_group);
    //    sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);

    //    uniforms = { &sdf_globals.preview_stroke_uniform };
    //    wgpuBindGroupRelease(preview_stroke_bind_group);
    //    preview_stroke_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_evaluate_shader, 3);
    //}

    // Upload preview data, first the stoke and tehn the edit list, since we are storing it in a vector
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.preview_stroke_uniform.data), 0u, &preview.sculpt_model_idx, sizeof(uint32_t));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.preview_stroke_uniform.data), 4 * sizeof(uint32_t), (&preview.to_upload_stroke), sizeof(sGPUStroke));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.preview_stroke_uniform.data), 4 * sizeof(uint32_t) + sizeof(sGPUStroke), preview.to_upload_edit_list->data(), preview.to_upload_stroke.edit_count * sizeof(Edit));

}

void SculptManager::init_uniforms()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    const uint32_t zero_value = 0u;

    // GPU return data buffer
    sGPU_SculptResults defaults;
    read_results.gpu_results_read_buffer = webgpu_context->create_buffer(sizeof(sGPU_SculptResults), WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, &defaults, "Evaluation results read buffer");
    gpu_results_uniform.data = webgpu_context->create_buffer(sizeof(sGPU_SculptResults), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, &defaults, "Evaluation results buffer");
    gpu_results_uniform.binding = 0u;
    gpu_results_uniform.buffer_size = sizeof(sGPU_SculptResults);

    // Evaluation data

    // Stroke context uniform
    {
        stroke_context_list_size = STROKE_CONTEXT_INITIAL_SIZE;
        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sGPUStroke) * STROKE_CONTEXT_INITIAL_SIZE;
        stroke_context_list.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        stroke_context_list.binding = 6;
        stroke_context_list.buffer_size = stroke_history_size;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), 0, &zero_value, sizeof(uint32_t));
    }

    // Context edit list
    {
        octree_edit_list_size = EDIT_BUFFER_INITIAL_SIZE;
        size_t edit_list_size = sizeof(Edit) * octree_edit_list_size;
        octree_edit_list.data = webgpu_context->create_buffer(edit_list_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree_edit_list");
        octree_edit_list.binding = 7;
        octree_edit_list.buffer_size = edit_list_size;
    }

    // Evaluator stroke culling buffers
    {
        uint32_t culling_size = sizeof(uint32_t) * (2u * sdf_globals.octree_last_level_size * MAX_STROKE_INFLUENCE_COUNT);
        stroke_culling_buffer.data = webgpu_context->create_buffer(culling_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_culling_data");
        stroke_culling_buffer.binding = 9;
        stroke_culling_buffer.buffer_size = culling_size;
    }

    // Evaluator work queues
    {
        WGPUBuffer octant_usage_buffers[2];

        // Ping pong buffers for read & write octants for the octree compute
        for (int i = 0; i < 2; ++i) {
            octant_usage_buffers[i] = webgpu_context->create_buffer(sdf_globals.octants_max_size * sizeof(uint32_t), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octant_usage");
            webgpu_context->update_buffer(octant_usage_buffers[i], 0, &zero_value, sizeof(uint32_t));
        }

        for (int i = 0; i < 4; ++i) {
            octant_usage_ping_pong_uniforms[i].data = octant_usage_buffers[i / 2];
            octant_usage_ping_pong_uniforms[i].binding = i % 2;
            octant_usage_ping_pong_uniforms[i].buffer_size = sdf_globals.octants_max_size * sizeof(uint32_t);
        }

        // Create another uniforms, same buffer, but diferent bidings, for initialization
        octant_usage_initialization_uniform[0].data = octant_usage_ping_pong_uniforms[0].data;
        octant_usage_initialization_uniform[0].binding = 1;
        octant_usage_initialization_uniform[0].buffer_size = octant_usage_ping_pong_uniforms[0].buffer_size;

        octant_usage_initialization_uniform[1].data = octant_usage_ping_pong_uniforms[2].data;
        octant_usage_initialization_uniform[1].binding = 2;
        octant_usage_initialization_uniform[1].buffer_size = octant_usage_ping_pong_uniforms[1].buffer_size;
    }

    // Intersect uniforms
    {
        ray_info_uniform.data = webgpu_context->create_buffer(sizeof(sGPU_RayData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "ray uniform");
        ray_info_uniform.binding = 0u;
        ray_info_uniform.buffer_size = sizeof(sGPU_RayData);

        sGPU_RayIntersection intialization;
        ray_intersection_info_uniform.data = webgpu_context->create_buffer(sizeof(sGPU_RayIntersection), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, &intialization, "Ray intersection result to copy");
        ray_intersection_info_uniform.binding = 0u;
        ray_intersection_info_uniform.buffer_size = sizeof(sGPU_RayIntersection);

        ray_sculpt_instances_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4u, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, nullptr, "Sculpt index counter");
        ray_sculpt_instances_uniform.binding = 1u;
        ray_sculpt_instances_uniform.buffer_size = sizeof(uint32_t) * 4;
    }

    // TODO: compute AABB per sculpt
    //// AABB sculpt compute
    //{
    //    // ((vec3 + padd) * 2) -> AABB struct size * num of workgroups
    //    const size_t aabb_buffer_size = sizeof(glm::vec4) * 2.0f * sdf_globals.octree_last_level_size / (8u * 8u * 8u);
    //    aabb_calculation_temp_buffer.data = webgpu_context->create_buffer(aabb_buffer_size, WGPUBufferUsage_Storage, nullptr, "AABB sculpt calculation buffer");
    //    aabb_calculation_temp_buffer.binding = 0u;
    //    aabb_calculation_temp_buffer.buffer_size = aabb_buffer_size;
    //}
}

void SculptManager::init_shaders()
{
    evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");
    brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    brick_copy_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy.wgsl");
    evaluation_initialization_shader = RendererStorage::get_shader("data/shaders/octree/initialization.wgsl");
    brick_unmark_shader = RendererStorage::get_shader("data/shaders/octree/brick_unmark.wgsl");
    sculpt_delete_shader = RendererStorage::get_shader("data/shaders/octree/sculpture_delete.wgsl");
    ray_intersection_result_and_clean_shader = RendererStorage::get_shader("data/shaders/octree/octree_ray_intersection_clean.wgsl");
    ray_intersection_shader = RendererStorage::get_shader("data/shaders/octree/octree_ray_intersection.wgsl");
}

void SculptManager::init_pipelines_and_bindgroups()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();


    // GPU return data
    {
        std::vector<Uniform*> uniforms = { &gpu_results_uniform };
        gpu_results_bindgroup = webgpu_context->create_bind_group(uniforms, ray_intersection_result_and_clean_shader, 1);
    }


    // Create bindgroups
    {
        std::vector<Uniform*> uniforms = {&octree_edit_list,
                                       &stroke_context_list, &sdf_globals.brick_buffers,&stroke_culling_buffer };
        evaluate_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 0);
    }

    // Brick removal pass
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.brick_buffers };
        indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, brick_removal_shader, 0);
    }

    // Octree increment iteration pass
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.indirect_buffers, &sdf_globals.brick_buffers };
        increment_level_bind_group = webgpu_context->create_bind_group(uniforms, increment_level_shader, 0);
    }

    // Octant usage, for propagating worl on the evalutor
    {
        for (int i = 0; i < 2; ++i) {
            std::vector<Uniform*> uniforms = { &octant_usage_ping_pong_uniforms[i], &octant_usage_ping_pong_uniforms[3 - i] }; // im sorry
            octant_usage_ping_pong_bind_groups[i] = webgpu_context->create_bind_group(uniforms, evaluate_shader, 2);
        }
    }

    // Write to texture
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.sdf_texture_uniform, &octree_edit_list, &stroke_culling_buffer,
                                           &stroke_context_list, &sdf_globals.brick_buffers, &sdf_globals.sdf_material_texture_uniform };
        write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);
    }

    // SDF atlases and samples bindgroup
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.sdf_texture_uniform, &sdf_globals.sdf_material_texture_uniform, &sdf_globals.linear_sampler_uniform };
        sdf_atlases_sampler_bindgroup = webgpu_context->create_bind_group(uniforms, ray_intersection_shader, 3);
    }

    // Octree initialiation bindgroup
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.indirect_buffers, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                            &sdf_globals.brick_buffers, &stroke_culling_buffer, &stroke_context_list };

        evaluation_initialization_bind_group = webgpu_context->create_bind_group(uniforms, evaluation_initialization_shader, 0);
    }

    // Sculpt delete
    {
        Uniform alt_brick_uniform = sdf_globals.brick_buffers;
        alt_brick_uniform.binding = 0u;

        std::vector<Uniform*> uniforms = { &alt_brick_uniform };
        sculpt_delete_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_delete_shader, 1u);
    }

    // Preview data bindgroup
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.preview_stroke_uniform };
        preview_stroke_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 3);
    }

    // Brick unmarking bindgroup
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.brick_buffers };
        brick_unmark_bind_group = webgpu_context->create_bind_group(uniforms, brick_unmark_shader, 0);
    }

    // Ray info bindgroup
    {
        std::vector<Uniform*> uniforms = { &ray_info_uniform, &ray_sculpt_instances_uniform, &rooms_renderer->get_global_sculpts_instance_data()};
        ray_sculpt_info_bind_group = webgpu_context->create_bind_group(uniforms, ray_intersection_shader, 1u);
    }

    // Ray intersection bindgroup
    {
        std::vector<Uniform*> uniforms = { &ray_intersection_info_uniform };
        ray_intersection_info_bind_group = webgpu_context->create_bind_group(uniforms, ray_intersection_shader, 0u);
    }

    // Create pipelines
    {
        evaluate_pipeline.create_compute_async(evaluate_shader);
        increment_level_pipeline.create_compute_async(increment_level_shader);
        write_to_texture_pipeline.create_compute_async(write_to_texture_shader);
        brick_removal_pipeline.create_compute_async(brick_removal_shader);
        brick_copy_pipeline.create_compute_async(brick_copy_shader);
        evaluation_initialization_pipeline.create_compute_async(evaluation_initialization_shader);
        brick_unmark_pipeline.create_compute_async(brick_unmark_shader);
        sculpt_delete_pipeline.create_compute_async(sculpt_delete_shader);
        ray_intersection_pipeline.create_compute_async(ray_intersection_shader);
        ray_intersection_result_and_clean_pipeline.create_compute_async(ray_intersection_result_and_clean_shader);

    }
}

void SculptManager::read_GPU_results()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

    read_results.map_in_progress = true;

    wgpuBufferMapAsync(read_results.gpu_results_read_buffer, WGPUMapMode_Read, 0u, sizeof(sGPU_SculptResults), get_mapped_result_buffer, (void*)&read_results);

    while (read_results.map_in_progress) {
        wgpuDeviceTick(webgpu_context->device);
    }

    wgpuBufferUnmap(read_results.gpu_results_read_buffer);

    performed_evaluation = false;
}

void get_mapped_result_buffer(WGPUBufferMapAsyncStatus status, void* user_payload)
{
    SculptManager::sGPU_ReadResults* result = (SculptManager::sGPU_ReadResults*)(user_payload);

    result->map_in_progress = false;

    if (status != WGPUBufferMapAsyncStatus_Success) {
        return;
    }

    size_t size = sizeof(sGPU_SculptResults);
    const void* gpu_buffer = wgpuBufferGetConstMappedRange(result->gpu_results_read_buffer, 0, size);
    memcpy_s(&result->loaded_results, size, gpu_buffer, size);

    /*if (result->loaded_results.ray_intersection.has_intersected == 1u) {
        Node::emit_signal("@on_gpu_intersection_results", (void*)result);
    }*/

    Node::emit_signal("@on_gpu_results", (void*)result);
}
