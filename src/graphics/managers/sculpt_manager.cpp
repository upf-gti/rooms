#include "sculpt_manager.h"

#include "rooms_includes.h"

#include "graphics/shader.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "framework/resources/sculpt.h"

#include <spdlog/spdlog.h>

void SculptManager::init()
{
    init_shaders();
    init_uniforms();
    init_pipelines_and_bindgroups();
}

void SculptManager::clean()
{
#ifndef DISABLE_RAYMARCHER
    wgpuBindGroupRelease(evaluate_bind_group);
    wgpuBindGroupRelease(increment_level_bind_group);
    wgpuBindGroupRelease(write_to_texture_bind_group);
    wgpuBindGroupRelease(octant_usage_ping_pong_bind_groups[0]);
    wgpuBindGroupRelease(octant_usage_ping_pong_bind_groups[1]);
    wgpuBindGroupRelease(preview_stroke_bind_group);

    delete evaluate_shader;
    delete increment_level_shader;
    delete write_to_texture_shader;
#endif
}

void SculptManager::update(WGPUCommandEncoder command_encoder)
{
    performed_evaluation = false;

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

    // Sculpture deleting and cleaning
//    {
//        if (sculpts_to_clean.size() > 0u) {
//            for (uint32_t i = 0u; i < sculpts_to_clean.size(); i++) {
//                sculpts_to_clean[i]->get_octree_uniform().destroy();
//                wgpuBindGroupRelease(sculpts_to_clean[i]->get_octree_bindgroup());
//            }
//            sculpts_to_clean.clear();
//        }
//
//        if (sculpts_to_delete.size() > 0u) {
//#ifndef NDEBUG
//            wgpuComputePassEncoderPushDebugGroup(compute_pass, "Sculpt removal");
//#endif
//            for (uint32_t i = 0u; i < sculpts_to_delete.size(); i++) {
//                compute_delete_sculpts(compute_pass, sculpts_to_delete[i]);
//                sculpts_to_clean.push_back(sculpts_to_delete[i]);
//            }
//            sculpts_to_delete.clear();
//#ifndef NDEBUG
//            wgpuComputePassEncoderPopDebugGroup(compute_pass);
//#endif
//        }
//    }

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    if (sculpts_to_create.size() > 0u) {
        sculpts_to_create.pop_back();
    }
}

void SculptManager::update_sculpt(Sculpt* sculpt, const sStrokeInfluence& strokes_to_process, const std::vector<Edit>& edits_to_process)
{
    evaluations_to_process.push_back({ sculpt, strokes_to_process, edits_to_process });
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

    std::vector<Uniform*> uniforms = { &octree_uniform };
    WGPUBindGroup octree_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, evaluate_shader, 1);

    //sculpt_instance_count.push_back(1u);

    return new Sculpt(sculpt_count++, octree_uniform, octree_buffer_bindgroup);
}

Sculpt* SculptManager::create_sculpt_from_history(const std::vector<Stroke>& stroke_history)
{
    Sculpt* new_sculpt = create_sculpt();
    new_sculpt->set_stroke_history(stroke_history);

    sculpts_to_create.push_back(new_sculpt);

    return new_sculpt;
}


void SculptManager::delete_sculpt(WGPUComputePassEncoder compute_pass, Sculpt* to_delete)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    sculpt_delete_pipeline.set(compute_pass);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, to_delete->get_octree_bindgroup(), 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt_delete_bindgroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octree_last_level_size / (8u * 8u * 8u), 1, 1);
}

void SculptManager::evaluate(WGPUComputePassEncoder compute_pass, const sEvaluateRequest& evaluate_request)
{
    if (!evaluate_shader || !evaluate_shader->is_loaded()) return;

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    spdlog::info("Evaluating sculpt {} with a context size of {}", evaluate_request.sculpt->get_sculpt_id(), evaluate_request.strokes_to_process.stroke_count);

    // Prepare for evaluation
    // Reset the brick instance counter
    uint32_t zero = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.brick_buffers.data), sizeof(uint32_t), &(zero), sizeof(uint32_t));

    // TODO uplad the AABB
    webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), 0, &evaluate_request.strokes_to_process, sizeof(uint32_t) * 4 * 4);

    // TODO: review undo
    uint32_t set_as_preview = /*(needs_undo) ? 0x01u :*/ 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(evaluate_request.sculpt->get_octree_uniform().data), sizeof(uint32_t) * 2u, &set_as_preview, sizeof(uint32_t));

    //spdlog::info("Evaluate stroke! id: {}, stroke context count: {}", stroke_to_compute->in_frame_stroke.stroke_id, stroke_to_compute->in_frame_influence.stroke_count);

    // TODO: debug render eval AABB

    // Upload strokes
    upload_strokes_and_edits(evaluate_request.strokes_to_process.strokes, evaluate_request.edit_to_process);

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
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);

            //uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview_stroke_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            int ping_pong_idx = 0;

            for (int j = 0; j <= sdf_globals.octree_depth; ++j) {

                evaluate_pipeline.set(compute_pass);

                wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluate_bind_group, 0, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 1, evaluate_request.sculpt->get_octree_bindgroup(), 0u, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_ping_pong_bind_groups[ping_pong_idx], 0, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

                wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 12u);

                increment_level_pipeline.set(compute_pass);

                wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
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
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, brick_copy_bind_group, 0, nullptr);
        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, sdf_globals.octants_max_size / (8u * 8u * 8u), 1, 1);

        increment_level_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    }

    performed_evaluation = true;
}

void SculptManager::evaluate_preview(WGPUComputePassEncoder compute_pass, const Stroke& stroke_to_preview)
{

}

void SculptManager::upload_strokes_and_edits(const std::vector<sToUploadStroke>& strokes_to_compute, const std::vector<Edit>& edits_to_upload)
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

        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * stroke_context_list_size;
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


        uniforms = { &merge_data_uniform, &octree_edit_list,
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
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_edit_list.data), 0, edits_to_upload.data(), sizeof(Edit) * edits_to_upload.size());
    webgpu_context->update_buffer(std::get<WGPUBuffer>(stroke_context_list.data), sizeof(uint32_t) * 4 * 4, strokes_to_compute.data(), sizeof(sToUploadStroke) * strokes_to_compute.size());

    //spdlog::info("min aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_min.x, stroke_to_compute->in_frame_influence.eval_aabb_min.y, stroke_to_compute->in_frame_influence.eval_aabb_min.z);
    //spdlog::info("max aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_max.x, stroke_to_compute->in_frame_influence.eval_aabb_max.y, stroke_to_compute->in_frame_influence.eval_aabb_max.z);
}


void SculptManager::init_uniforms()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    const uint32_t zero_value = 0u;

    // Evaluation data struct
    // Edit count & other merger data
    merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "merge_data");
    merge_data_uniform.binding = 1;
    merge_data_uniform.buffer_size = sizeof(sMergeData);


    // Stroke context uniform
    {
        stroke_context_list_size = STROKE_CONTEXT_INITIAL_SIZE;
        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * STROKE_CONTEXT_INITIAL_SIZE;
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
}

void SculptManager::init_shaders()
{
    evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");
    brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    brick_copy_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy.wgsl");
    evaluation_initialization_shader = RendererStorage::get_shader("data/shaders/octree/initialization.wgsl");
    //brick_unmark_shader = RendererStorage::get_shader("data/shaders/octree/brick_unmark.wgsl");
    sculpt_delete_shader = RendererStorage::get_shader("data/shaders/octree/sculpture_delete.wgsl");
}

void SculptManager::init_pipelines_and_bindgroups()
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    // Create bindgroups
    {
        std::vector<Uniform*> uniforms = { &merge_data_uniform, &octree_edit_list,
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

    // Brick copy for brick instanced rendering
    {
        std::vector<Uniform*> uniforms = { &sdf_globals.brick_copy_buffer, &sdf_globals.brick_buffers };
        brick_copy_bind_group = webgpu_context->create_bind_group(uniforms, brick_copy_shader, 0);
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

    // Create pipelines
    {
        evaluate_pipeline.create_compute_async(evaluate_shader);
        increment_level_pipeline.create_compute_async(increment_level_shader);
        write_to_texture_pipeline.create_compute_async(write_to_texture_shader);
        brick_removal_pipeline.create_compute_async(brick_removal_shader);
        brick_copy_pipeline.create_compute_async(brick_copy_shader);
        evaluation_initialization_pipeline.create_compute_async(evaluation_initialization_shader);
        //brick_unmark_pipeline.create_compute_async(compute_octree_brick_unmark_shader);
        sculpt_delete_pipeline.create_compute_async(sculpt_delete_shader);
    }
}
