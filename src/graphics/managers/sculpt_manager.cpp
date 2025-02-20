#include "sculpt_manager.h"

#include "rooms_includes.h"

#include "engine/rooms_engine.h"

#include "graphics/shader.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "framework/resources/sculpt.h"
#include "framework/nodes/sculpt_node.h"

#include <spdlog/spdlog.h>

void get_mapped_result_buffer(WGPUMapAsyncStatus status, void* user_payload);

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

    wgpuBindGroupRelease(gpu_results_bindgroup);
    wgpuBindGroupRelease(ray_sculpt_info_bind_group);
    wgpuBindGroupRelease(ray_intersection_info_bind_group);
    wgpuBindGroupRelease(sdf_atlases_sampler_bindgroup);
    wgpuBindGroupRelease(indirect_brick_removal_bind_group);
    wgpuBindGroupRelease(sculpt_delete_bindgroup);
    wgpuBindGroupRelease(brick_unmark_bind_group);
    wgpuBindGroupRelease(evaluator_preview_bind_group);
    wgpuBindGroupRelease(evaluator_aabb_culling_step_bind_group);
    wgpuBindGroupRelease(evaluator_interval_culling_step_bind_group);
    wgpuBindGroupRelease(evaluator_write_to_texture_setup_bind_group);
    wgpuBindGroupRelease(evaluator_write_to_texture_step_bind_group);

    // todo: THIS CRASHES
    evaluation_job_result_count_uniform.destroy();
    evaluation_aabb_culling_count_uniform.destroy();
    evaluation_culling_dispatch_uniform.destroy();
    evaluator_num_bricks_by_wg_uniform.destroy();
    evaluation_write_to_tex_buffer_alt_uniform.destroy();
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

    delete write_to_texture_shader;
    delete brick_removal_shader;
    delete brick_copy_aabb_gen_shader;
    delete brick_unmark_shader;
    delete sculpt_delete_shader;
    delete ray_intersection_shader;
    delete ray_intersection_result_and_clean_shader;
#endif
}

void SculptManager::update(WGPUCommandEncoder command_encoder)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    // New render pass for the interseccions
    if (intersections_to_compute > 0u) {
        WGPUComputePassDescriptor compute_pass_desc = {};

        std::vector<WGPUPassTimestampWrites> timestampWrites(1);
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

    std::vector<WGPUPassTimestampWrites> timestampWrites(1);
    timestampWrites[0].beginningOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "pre_evaluation");
    timestampWrites[0].querySet = Renderer::instance->get_query_set();
    timestampWrites[0].endOfPassWriteIndex = Renderer::instance->timestamp(command_encoder, "evaluation");

    compute_pass_desc.timestampWrites = timestampWrites.data();

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    // TODO: only do this when necessary
    // Clean preview indirect
    uint32_t zero = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.brick_buffers.data), sizeof(uint32_t) * 2u, &zero, sizeof(uint32_t));
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 5u, &zero, sizeof(uint32_t));

    if (!evaluations_to_process.empty()) {
        sEvaluateRequest& evaluate_request = evaluations_to_process.back();

        if (evaluate(compute_pass, evaluate_request)) {
            evaluations_to_process.pop_back();
        }
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
            webgpu_context->push_debug_group(compute_pass, { "Sculpt removal", WGPU_STRLEN });
#endif
            for (uint32_t i = 0u; i < sculpts_to_delete.size(); i++) {
                delete_sculpt(compute_pass, sculpts_to_delete[i]);
                sculpts_to_clean.push_back(sculpts_to_delete[i]);
            }
            sculpts_to_delete.clear();
#ifndef NDEBUG
            webgpu_context->pop_debug_group(compute_pass);
#endif
        }
    }

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    if (intersections_to_compute > 0u || has_performed_evaluation()) {
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

    AABB preview_aabb = preview.to_upload_stroke.get_world_AABB_of_edit_list(preview_edits);

    preview.to_upload_stroke.aabb_min = preview_aabb.center - preview_aabb.half_size;
    preview.to_upload_stroke.aabb_max = preview_aabb.center + preview_aabb.half_size;

    preview.needs_computing = true;
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
    sculpt_to_test = nullptr;
}

void SculptManager::set_ray_to_test(const glm::vec3& ray_origin, const glm::vec3& ray_dir, Sculpt* sculpt, const uint32_t model_id) {
    if (intersections_to_compute > 0u) {
        spdlog::error("Only one ray test per frame!");
        assert(0u);
    }

    ray_to_upload.ray_origin = ray_origin;
    ray_to_upload.ray_direction = ray_dir;
    intersections_to_compute = 1u;
    model_to_test_idx = model_id;
    sculpt_to_test = sculpt;
    intersection_node_to_test = nullptr;
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
    WGPUBindGroup evaluate_sculpt_bindgroup = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader("data/shaders/octree/prepare_indirect_sculpt_render.wgsl"), 1u, "Write read octree bindgroup");

    uniforms = { &octree_uniform };
    WGPUBindGroup octree_bindgroup = webgpu_context->create_bind_group(uniforms, evaluator_2_interval_culling_step_shader, 1u, "Octree bindgroup");

    uniforms = { &brick_index_buffer, &indirect_buffer };
    WGPUBindGroup readonly_sculpt_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl"), 2u, "Read only octree bindgroup");


    uniforms = { &octree_uniform, &brick_index_buffer, &indirect_buffer, &sdf_globals.brick_buffers};
    WGPUBindGroup brick_copy_aabb_gen_bind_group = webgpu_context->create_bind_group(uniforms, brick_copy_aabb_gen_shader, 0u, "brick copy bindgroup");

    Sculpt* new_sculpt = new Sculpt(sculpt_count++, octree_uniform, indirect_buffer, brick_index_buffer, octree_bindgroup);

    new_sculpt->set_brick_copy_bindgroup(brick_copy_aabb_gen_bind_group);
    new_sculpt->set_readonly_octree_bindgroup(readonly_sculpt_buffer_bindgroup);
    new_sculpt->set_sculpt_evaluation_bindgroup(evaluate_sculpt_bindgroup);

    return new_sculpt;
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

    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

#ifndef NDEBUG
    webgpu_context->push_debug_group(compute_pass, { "Delete Sculpt", WGPU_STRLEN });
#endif
    sculpt_delete_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, to_delete->get_octree_bindgroup(), 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_delete_bindgroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octree_last_level_size / (8u * 8u * 8u), 1, 1);
#ifndef NDEBUG
    webgpu_context->pop_debug_group(compute_pass);
#endif
}

bool SculptManager::evaluate(WGPUComputePassEncoder compute_pass, const sEvaluateRequest& evaluate_request)
{
    if (!evaluator_1_aabb_culling_step_pipeline.is_loaded() ||
        !evaluator_1_5_interval_culling_step_pipeline.is_loaded() ||
        !evaluator_2_interval_culling_step_pipeline.is_loaded() ||
        !evaluator_2_5_write_to_texture_setup_pipeline.is_loaded() ||
        !write_to_texture_pipeline.is_loaded() ||
        !brick_copy_aabb_gen_pipeline.is_loaded()) {
        return false;
    }

    // Sculpt might have been deleted since the request was sent
    if (evaluate_request.sculpt->is_deleted()) {
        return true;
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

    // Upload strokes
    upload_strokes_and_edits(evaluate_request.strokes_to_process.stroke_count, evaluate_request.strokes_to_process.strokes, evaluate_request.edit_count, evaluate_request.edit_to_process);
    // Upload stroke count and stroke context AABB
    webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), 0, &evaluate_request.strokes_to_process, sizeof(glm::vec4) * 3);

    // Compute dispatches
    {
#ifndef NDEBUG
        webgpu_context->push_debug_group(compute_pass, { "SDF Evaluator", WGPU_STRLEN } );
#endif
        // prepare buffers
        uint32_t to_fill = 0u;
        uint32_t to_fill4 = 0u;
        int32_t to_fill2 = sdf_globals.octree_last_level_size;
        uint32_t to_fill3[3] = { 0u, 1u, 1u };
        webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluation_job_result_count_uniform.data), 0, &to_fill3, sizeof(uint32_t) * 2u);
        webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluation_aabb_culling_count_uniform.data), 0, &to_fill2, sizeof(int32_t));
        webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluation_culling_dispatch_uniform.data), 0, to_fill3, sizeof(int32_t) * 3u);

        webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluate_request.sculpt->get_indirect_render_buffer().data), sizeof(uint32_t), &to_fill, sizeof(uint32_t));
        webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluate_request.sculpt->get_indirect_render_buffer().data), sizeof(uint32_t) * 4u, &to_fill, sizeof(uint32_t));

        evaluator_1_aabb_culling_step_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluator_stroke_history_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluator_aabb_culling_step_bind_group, 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, (uint32_t)glm::ceil(sdf_globals.octree_last_level_size / 512.0), 1, 1);

        evaluator_1_5_interval_culling_step_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluator_aabb_culling_step_bind_group, 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

        evaluator_2_interval_culling_step_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluator_interval_culling_step_bind_group, 0u, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(evaluation_culling_dispatch_uniform.data), 0u);

        evaluator_2_5_write_to_texture_setup_pipeline.set(compute_pass);
        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluator_write_to_texture_setup_bind_group, 0u, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluator_write_to_texture_setup_bind_group, 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

#ifndef NDEBUG
        webgpu_context->push_debug_group(compute_pass, { "Write to texture", WGPU_STRLEN } );
#endif
        // Write to texture dispatch
        write_to_texture_pipeline.set(compute_pass);

        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluator_write_to_texture_step_bind_group, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(evaluation_culling_dispatch_uniform.data), 0u);

#ifndef NDEBUG
        webgpu_context->pop_debug_group(compute_pass);
#endif

        brick_copy_aabb_gen_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, evaluate_request.sculpt->get_brick_copy_aabb_gen_bindgroup(), 0u, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, gpu_results_bindgroup, 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octree_last_level_size / (512.0f), 1, 1);

        // Clean the texture atlas bricks dispatch
        brick_removal_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, indirect_brick_removal_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

#ifndef NDEBUG
        webgpu_context->pop_debug_group(compute_pass);
#endif

    }

    performed_evaluation = true;

    rooms_renderer->request_timestamps();

    return true;
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
    webgpu_context->push_debug_group(compute_pass, { "Brick unmark", WGPU_STRLEN });
#endif
    brick_unmark_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, brick_unmark_bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octants_max_size / (8u * 8u * 8u), 1, 1);
#ifndef NDEBUG
    webgpu_context->pop_debug_group(compute_pass);
#endif
    previus_dispatch_had_preview = false;
}

void SculptManager::evaluate_preview(WGPUComputePassEncoder compute_pass)
{
    if (!evaluator_preview_step_pipeline.is_loaded()) {
        return;
    }
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    upload_preview_strokes();

#ifndef NDEBUG
    webgpu_context->push_debug_group(compute_pass, { "Preview evaluation", WGPU_STRLEN });
#endif

    evaluator_preview_step_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, evaluator_preview_bind_group, 0u, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, preview.sculpt->get_octree_bindgroup(), 0u, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, (uint32_t)glm::ceil(sdf_globals.octree_last_level_size / 512.0), 1u, 1u);

#ifndef NDEBUG
    webgpu_context->pop_debug_group(compute_pass);
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
    webgpu_context->push_debug_group(compute_pass, { "Ray intersection", WGPU_STRLEN });
#endif

    // TODO: add range checks for the ray, for an early out
    // TODO: if bindless, we could do a workgroup/thread per ray

    // Upload ray uniform
    uint32_t starting_idx = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_info_uniform.data), 0u, &ray_to_upload, sizeof(sGPU_RayData));
    if (intersection_node_to_test != nullptr) {
        starting_idx = intersection_node_to_test->get_in_frame_render_instance_idx();
    }
    else if (sculpt_to_test != nullptr) {
        starting_idx = model_to_test_idx;
    }
    webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_sculpt_instances_uniform.data), 0u, &starting_idx, sizeof(uint32_t));


    ray_intersection_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, ray_intersection_info_bind_group, 0u, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, ray_sculpt_info_bind_group, 0u, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 3u, sdf_atlases_sampler_bindgroup, 0u, nullptr);

    if (intersection_node_to_test == nullptr) {
        for (auto& it : rooms_renderer->get_sculpts_render_list()) {
            Sculpt* curr_sculpt = it.second.sculpt;
            const uint32_t instances_count = it.second.instance_count;

            wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, curr_sculpt->get_octree_bindgroup(), 0u, nullptr);

            for (uint32_t i = 0u; i < instances_count; i++) {
                wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);
            }
        }
    }
    else {
        Sculpt* curr_sculpt = (sculpt_to_test) ? sculpt_to_test : intersection_node_to_test->get_sculpt_data();

        wgpuComputePassEncoderSetBindGroup(compute_pass, 2u, curr_sculpt->get_octree_bindgroup(), 0u, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);
    }

    ray_intersection_result_and_clean_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, gpu_results_bindgroup, 0u, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

#ifndef NDEBUG
    webgpu_context->pop_debug_group(compute_pass);
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
                                                &stroke_context_list, &sdf_globals.brick_buffers, &sdf_globals.sdf_material_texture_uniform ,
                                                &evaluator_num_bricks_by_wg_uniform, &evaluation_job_result_count_uniform, &evaluation_write_to_tex_buffer_alt_uniform };

        wgpuBindGroupRelease(evaluator_write_to_texture_step_bind_group);
        evaluator_write_to_texture_step_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);

        uniforms = { &stroke_context_list, &stroke_culling_buffer };
        wgpuBindGroupRelease(evaluator_stroke_history_bind_group);
        evaluator_stroke_history_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_1_aabb_culling_step_shader, 0u);

        uniforms = { &stroke_culling_buffer, &octree_edit_list, &stroke_context_list, &sdf_globals.brick_buffers, &evaluation_job_result_count_uniform, &octant_usage_ping_pong_uniforms[1] , &evaluation_write_to_tex_buffer_uniform };
        wgpuBindGroupRelease(evaluator_interval_culling_step_bind_group);
        evaluator_interval_culling_step_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_2_interval_culling_step_shader, 0u);
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
    //    std::vector<Uniform*> uniforms = { &sculpts_instance_data_uniform, &linear_sampler_uniform, &sdf_texture_uniform, &octree_brick_buffers, &octree_brick_copy_aabb_gen_buffer, &sdf_material_texture_uniform, &preview_stroke_uniform_2 };
    //    wgpuBindGroupRelease(render_proxy_geometry_bind_group);
    //    render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);

    //    uniforms = { &sculpts_instance_data_uniform, &preview_stroke_uniform_2, &octree_brick_buffers };
    //    wgpuBindGroupRelease(sculpt_data_bind_preview_group);
    //    sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);

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
        uint32_t culling_size = sizeof(uint32_t) * (sdf_globals.octree_last_level_size * MAX_STROKE_INFLUENCE_COUNT);
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

        sGPU_RayIntersectionData intialization;
        ray_intersection_info_uniform.data = webgpu_context->create_buffer(sizeof(sGPU_RayIntersectionData), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, &intialization, "Ray intersection result to copy");
        ray_intersection_info_uniform.binding = 0u;
        ray_intersection_info_uniform.buffer_size = sizeof(sGPU_RayIntersectionData);

        ray_sculpt_instances_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4u, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, nullptr, "Sculpt index counter");
        ray_sculpt_instances_uniform.binding = 1u;
        ray_sculpt_instances_uniform.buffer_size = sizeof(uint32_t) * 4;
    }

    // New Evaluator passes uniforms
    {
        evaluation_aabb_culling_count_uniform.data = webgpu_context->create_buffer(sizeof(int32_t), WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, nullptr, "AABB count");
        evaluation_aabb_culling_count_uniform.binding = 0u;
        evaluation_aabb_culling_count_uniform.buffer_size = sizeof(int32_t);

        evaluation_job_result_count_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 2u, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, nullptr, "Job result count");
        evaluation_job_result_count_uniform.binding = 2u;
        evaluation_job_result_count_uniform.buffer_size = sizeof(uint32_t) * 2u;

        evaluation_culling_dispatch_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 3u, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect, nullptr, "Culling indirect buffer counter");
        evaluation_culling_dispatch_uniform.binding = 3u;
        evaluation_culling_dispatch_uniform.buffer_size = sizeof(uint32_t) * 3u;

        evaluation_culling_dispatch_alt_uniform = evaluation_culling_dispatch_uniform;
        evaluation_culling_dispatch_alt_uniform.binding = 4u;


        evaluation_write_to_tex_buffer_uniform = octant_usage_ping_pong_uniforms[2];
        evaluation_write_to_tex_buffer_uniform.binding = 3u;

        evaluation_write_to_tex_buffer_alt_uniform = evaluation_write_to_tex_buffer_uniform;
        evaluation_write_to_tex_buffer_alt_uniform.binding = 4u;

        evaluator_num_bricks_by_wg_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t), WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, nullptr, "Num briks per write 2 tex dispatch");
        evaluator_num_bricks_by_wg_uniform.binding = 0u;
        evaluator_num_bricks_by_wg_uniform.buffer_size = sizeof(uint32_t);
    }
}

void SculptManager::init_shaders()
{
    evaluator_1_aabb_culling_step_shader = RendererStorage::get_shader("data/shaders/octree/evaluator_1_aabb_culling_step.wgsl");
    evaluator_1_5_interval_culling_step_shader = RendererStorage::get_shader("data/shaders/octree/evaluator_1-5_interval_setup.wgsl");
    evaluator_2_interval_culling_step_shader = RendererStorage::get_shader("data/shaders/octree/evaluator_2_interval_culling_step.wgsl");
    evaluator_2_5_write_to_texture_setup_step_shader = RendererStorage::get_shader("data/shaders/octree/evaluator_2-5_write_to_texture_step.wgsl");
    evaluator_preview_step_shader = RendererStorage::get_shader("data/shaders/octree/evaluator_preview_culling.wgsl");

    write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");
    brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    brick_copy_aabb_gen_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy_aabb_gen.wgsl");
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

    // Brick removal pass
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.brick_buffers };
        indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, brick_removal_shader, 0);
    }

    // SDF atlases and samples bindgroup
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.sdf_texture_uniform, &sdf_globals.sdf_material_texture_uniform, &sdf_globals.linear_sampler_uniform };
        sdf_atlases_sampler_bindgroup = webgpu_context->create_bind_group(uniforms, ray_intersection_shader, 3);
    }

    // Sculpt delete
    {
        Uniform alt_brick_uniform = sdf_globals.brick_buffers;
        alt_brick_uniform.binding = 0u;

        std::vector<Uniform*> uniforms = { &alt_brick_uniform };
        sculpt_delete_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_delete_shader, 1u);
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

    // Evaluator pipelines
    {
        // Coarse AABB culling
        {
            std::vector<Uniform*> uniforms = { &evaluation_job_result_count_uniform, &evaluation_aabb_culling_count_uniform, &octant_usage_ping_pong_uniforms[1], &evaluation_culling_dispatch_uniform };
            evaluator_aabb_culling_step_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_1_aabb_culling_step_shader, 1u);

            uniforms = { &stroke_context_list, &stroke_culling_buffer };
            evaluator_stroke_history_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_1_aabb_culling_step_shader, 0u);

            uniforms = { &stroke_culling_buffer, &octree_edit_list, &stroke_context_list, &sdf_globals.brick_buffers, &evaluation_job_result_count_uniform, &octant_usage_ping_pong_uniforms[1] , &evaluation_write_to_tex_buffer_uniform };
            evaluator_interval_culling_step_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_2_interval_culling_step_shader, 0u);
        }

        {
            std::vector<Uniform*> uniforms = { &evaluation_job_result_count_uniform, &sdf_globals.sdf_texture_uniform, &octree_edit_list, &stroke_culling_buffer,
                                                &stroke_context_list, &sdf_globals.brick_buffers, &sdf_globals.sdf_material_texture_uniform ,
                                                &evaluator_num_bricks_by_wg_uniform, &evaluation_write_to_tex_buffer_alt_uniform };
            evaluator_write_to_texture_step_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0u);

            uniforms = { &evaluation_job_result_count_uniform, &evaluation_culling_dispatch_alt_uniform, &evaluator_num_bricks_by_wg_uniform, &evaluation_write_to_tex_buffer_uniform };
            evaluator_write_to_texture_setup_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_2_5_write_to_texture_setup_step_shader, 0u);
        }

        // Preview bd
        {
            std::vector<Uniform*> uniforms = { &sdf_globals.preview_stroke_uniform, &sdf_globals.brick_buffers, &sdf_globals.indirect_buffers };
            evaluator_preview_bind_group = webgpu_context->create_bind_group(uniforms, evaluator_preview_step_shader, 0u);
        }
    }

    // Create pipelines
    {
        write_to_texture_pipeline.create_compute_async(write_to_texture_shader);
        brick_removal_pipeline.create_compute_async(brick_removal_shader);
        brick_copy_aabb_gen_pipeline.create_compute_async(brick_copy_aabb_gen_shader);
        brick_unmark_pipeline.create_compute_async(brick_unmark_shader);
        sculpt_delete_pipeline.create_compute_async(sculpt_delete_shader);
        ray_intersection_pipeline.create_compute_async(ray_intersection_shader);
        ray_intersection_result_and_clean_pipeline.create_compute_async(ray_intersection_result_and_clean_shader);

        evaluator_1_aabb_culling_step_pipeline.create_compute_async(evaluator_1_aabb_culling_step_shader);
        evaluator_2_interval_culling_step_pipeline.create_compute_async(evaluator_2_interval_culling_step_shader);
        evaluator_1_5_interval_culling_step_pipeline.create_compute_async(evaluator_1_5_interval_culling_step_shader);
        evaluator_2_5_write_to_texture_setup_pipeline.create_compute_async(evaluator_2_5_write_to_texture_setup_step_shader);
        evaluator_preview_step_pipeline.create_compute_async(evaluator_preview_step_shader);
    }
}

void SculptManager::read_GPU_results()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    uint32_t frame_counter = rooms_renderer->get_frame_counter();

    if (!reading_gpu_results && (frame_counter - frame_of_last_gpu_read > 10)) {
        reading_gpu_results = true;
        frame_of_last_gpu_read = frame_counter;

        webgpu_context->read_buffer_async(std::get<WGPUBuffer>(gpu_results_uniform.data), sizeof(sGPU_SculptResults), [&](const void* output_buffer, void* user_data) {
            memcpy(&loaded_results, output_buffer, sizeof(sGPU_SculptResults));
            Node::emit_signal("@on_gpu_results", (void*)&loaded_results);
            reading_gpu_results = false;
        }, nullptr);
    }

    performed_evaluation = false;
}

void SculptManager::apply_sculpt_offset(SculptNode* sculpt_node, const glm::vec3& texture_space_offset)
{
    Sculpt* gpu_data = sculpt_node->get_sculpt_data();
    std::vector<Stroke> stroke_history = gpu_data->get_stroke_history();

    for (auto& stroke : stroke_history) {

        for (int i = 0; i < stroke.edit_count; i++) {
            Edit& edit = stroke.edits[i];
            edit.position += texture_space_offset;
        }
    }

    sculpt_node->from_history(stroke_history, true);

    delete_sculpt(gpu_data);
}
