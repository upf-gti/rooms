#include "sculpt.h"

#include "graphics/shader.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include <spdlog/spdlog.h>

void SculptManager::init()
{
    // Size of penultimate level
    SDF_Globals.octants_max_size = pow(floorf(SDF_RESOLUTION / 10.0f), 3.0f);

    SDF_Globals.octree_depth = static_cast<uint8_t>(OCTREE_DEPTH);

    // total size considering leaves and intermediate levels
    SDF_Globals.octree_total_size = (pow(8, SDF_Globals.octree_depth + 1) - 1) / 7;

    SDF_Globals.octree_last_level_size = SDF_Globals.octree_total_size - (pow(8, SDF_Globals.octree_depth) - 1) / 7;

    uint32_t brick_count_in_axis = static_cast<uint32_t>(SDF_RESOLUTION / BRICK_SIZE);
    SDF_Globals.max_brick_count = brick_count_in_axis * brick_count_in_axis * brick_count_in_axis;

    SDF_Globals.empty_brick_and_removal_buffer_count = SDF_Globals.max_brick_count + (SDF_Globals.max_brick_count % 4);
    float octree_space_scale = powf(2.0, SDF_Globals.octree_depth + 3);

    // Scale the size of a brick
    Shader::set_custom_define("WORLD_SPACE_SCALE", octree_space_scale); // Worldspace scale is 1/octree_max_width
    Shader::set_custom_define("OCTREE_DEPTH", SDF_Globals.octree_depth);
    Shader::set_custom_define("OCTREE_TOTAL_SIZE", SDF_Globals.octree_total_size);
    Shader::set_custom_define("PREVIEW_PROXY_BRICKS_COUNT", PREVIEW_PROXY_BRICKS_COUNT);
    Shader::set_custom_define("BRICK_REMOVAL_COUNT", SDF_Globals.empty_brick_and_removal_buffer_count);
    Shader::set_custom_define("MAX_SUBDIVISION_SIZE", SDF_Globals.octree_last_level_size);
    Shader::set_custom_define("MAX_STROKE_INFLUENCE_COUNT", max_stroke_influence_count);

    SDF_Globals.brick_world_size = (SCULPT_MAX_SIZE / octree_space_scale) * 8.0f;
}

void SculptManager::clean()
{

}

Sculpt* SculptManager::create_sculpt()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    Uniform octree_uniform;
    std::vector<sOctreeNode> octree_default(SDF_Globals.octree_total_size + 1);
    octree_uniform.data = webgpu_context->create_buffer(sizeof(uint32_t) * 4 + SDF_Globals.octree_total_size * sizeof(sOctreeNode), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octree");
    octree_uniform.binding = 0;
    octree_uniform.buffer_size = sizeof(uint32_t) * 4u + SDF_Globals.octree_total_size * sizeof(sOctreeNode);
    // Set the id of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 3, &sculpt_count, sizeof(uint32_t));
    // Set default values of the octree
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_uniform.data), sizeof(uint32_t) * 4, octree_default.data(), sizeof(sOctreeNode) * SDF_Globals.octree_total_size);

    std::vector<Uniform*> uniforms = { &octree_uniform };
    WGPUBindGroup octree_buffer_bindgroup = webgpu_context->create_bind_group(uniforms, evaluate_shader, 1);

    //sculpt_instance_count.push_back(1u);

    return new Sculpt( sculpt_count++, octree_uniform, octree_buffer_bindgroup );
}


void SculptManager::delete_sculpt(WGPUComputePassEncoder compute_pass, const Sculpt& to_delete) {

}

void SculptManager::evaluate(WGPUComputePassEncoder compute_pass, const Sculpt& sculpt, const std::vector<sToUploadStroke>& stroke_to_eval, const std::vector<Edit>& edits_to_upload, const AABB& stroke_aabb)
{
    if (!evaluate_shader || !evaluate_shader->is_loaded()) return;

    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Prepare for evaluation
    // Reset the brick instance counter
    uint32_t zero = 0u;
    webgpu_context->update_buffer(std::get<WGPUBuffer>(SDF_Globals.brick_buffers.data), sizeof(uint32_t), &(zero), sizeof(uint32_t));

    // TODO uplad the AABB
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), 0, &stroke_to_compute->in_frame_influence, sizeof(uint32_t) * 4 * 4);

    // TODO: debug render eval AABB

    // Upload strokes
    upload_strokes_and_edits(stroke_to_eval, edits_to_upload);

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
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0u, nullptr);

            //uint32_t stroke_dynamic_offset = i * sizeof(Stroke);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, compute_stroke_buffer_bind_group, 1, &stroke_dynamic_offset);

            //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, preview_stroke_bind_group, 0, nullptr);

            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

            int ping_pong_idx = 0;

            for (int j = 0; j <= SDF_Globals.octree_depth; ++j) {

                evaluate_pipeline.set(compute_pass);

                wgpuComputePassEncoderSetBindGroup(compute_pass, 0, evaluate_bind_group, 0, nullptr);
                //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0u, nullptr);
                wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_ping_pong_bind_groups[ping_pong_idx], 0, nullptr);
                //wgpuComputePassEncoderSetBindGroup(compute_pass, 3, preview_stroke_bind_group, 0, nullptr);

                wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

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

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 12u);

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

            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_pass, std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), sizeof(uint32_t) * 8u);

        }

#ifndef NDEBUG
        wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

        brick_copy_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, brick_copy_bind_group, 0, nullptr);
        //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, sculpt.octree_bindgroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, SDF_Globals.octants_max_size / (8u * 8u * 8u), 1, 1);

        increment_level_pipeline.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, increment_level_bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    }
}



void SculptManager::upload_strokes_and_edits(const std::vector<sToUploadStroke> &strokes_to_compute, const std::vector<Edit> &edits_to_upload)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
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

    if (strokes_to_compute.size() > stroke_context_size) {
        spdlog::info("Resized GPU stroke context buffer from {} to {}", stroke_context_size, strokes_to_compute.size());

        stroke_context_size = strokes_to_compute.size();

        octree_stroke_context.destroy();

        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * stroke_context_size;
        octree_stroke_context.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        octree_stroke_context.binding = 6;
        octree_stroke_context.buffer_size = stroke_history_size;

        recreated_stroke_context_buffer = true;
    }

    // If one of the buffers was recreated/resized, then recreate the necessary bindgroups
    if (recreated_stroke_context_buffer || recreated_edit_buffer) {
        std::vector<Uniform*> uniforms = { &SDF_Globals.sdf_texture_uniform, &octree_edit_list, &stroke_culling_data,
                                           &octree_stroke_context, & SDF_Globals.brick_buffers, & SDF_Globals.sdf_material_texture_uniform };

        wgpuBindGroupRelease(write_to_texture_bind_group);
        write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);


        uniforms = { &merge_data_uniform, &octree_edit_list,
                     &octree_stroke_context, & SDF_Globals.brick_buffers, &stroke_culling_data };

        wgpuBindGroupRelease(evaluate_bind_group);
        evaluate_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 0);
    }

    if (recreated_stroke_context_buffer) {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                        &SDF_Globals.brick_buffers, &stroke_culling_data, &octree_stroke_context };

        evaluation_initialization_bind_group = webgpu_context->create_bind_group(uniforms, evaluation_initialization_shader, 0);
    }

    // Upload the data to the GPU
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_edit_list.data), 0, edits_to_upload.data(), sizeof(Edit) * stroke_manager.edit_list_count);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), sizeof(uint32_t) * 4 * 4, strokes_to_compute.data(), stroke_to_compute->in_frame_influence.stroke_count * sizeof(sToUploadStroke));

    //spdlog::info("min aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_min.x, stroke_to_compute->in_frame_influence.eval_aabb_min.y, stroke_to_compute->in_frame_influence.eval_aabb_min.z);
    //spdlog::info("max aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_max.x, stroke_to_compute->in_frame_influence.eval_aabb_max.y, stroke_to_compute->in_frame_influence.eval_aabb_max.z);
}



void SculptManager::init_uniforms() {
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    const uint32_t zero_value = 0u;

    // Evaluation data struct
    // Edit count & other merger data
    compute_merge_data_uniform.data = webgpu_context->create_buffer(sizeof(sMergeData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "merge_data");
    compute_merge_data_uniform.binding = 1;
    compute_merge_data_uniform.buffer_size = sizeof(sMergeData);


    // Stroke context uniform
    {
        stroke_context_size = STROKE_CONTEXT_INTIAL_SIZE;
        size_t stroke_history_size = sizeof(uint32_t) * 4u * 4u + sizeof(sToUploadStroke) * STROKE_CONTEXT_INTIAL_SIZE;
        octree_stroke_context.data = webgpu_context->create_buffer(stroke_history_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_history");
        octree_stroke_context.binding = 6;
        octree_stroke_context.buffer_size = stroke_history_size;
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), 0, &zero_value, sizeof(uint32_t));
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
        uint32_t culling_size = sizeof(uint32_t) * (2u * SDF_Globals.octree_last_level_size * max_stroke_influence_count);
        stroke_culling_data.data = webgpu_context->create_buffer(culling_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "stroke_culling_data");
        stroke_culling_data.binding = 9;
        stroke_culling_data.buffer_size = culling_size;
    }

    // Indirect dispatch buffer
    {
        uint32_t buffer_size = sizeof(uint32_t) * 4u * 4u;
        octree_indirect_buffer_struct.data = webgpu_context->create_buffer(buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect | WGPUBufferUsage_Storage, nullptr, "indirect_buffers_struct");
        octree_indirect_buffer_struct.binding = 8u;
        octree_indirect_buffer_struct.buffer_size = buffer_size;

        // Create another uniform, same buffer, different biding
        octree_indirect_buffer_struct_2 = octree_indirect_buffer_struct;
        octree_indirect_buffer_struct_2.binding = 7u;

        uint32_t default_indirect_values[16u] = {
            36u, 0u, 0u, 0u, // bricks indirect call
            36u, 0u, 0u, 0u,// preview bricks indirect call
            0u, 1u, 1u, 0u, // brick removal call (1 padding)
            1u, 1u, 1u, 0u // octree subdivision
        };
        webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_indirect_buffer_struct.data), 0, default_indirect_values, sizeof(uint32_t) * 16u);
    }

    // Evaluator work queues
    {
        WGPUBuffer octant_usage_buffers[2];

        // Ping pong buffers for read & write octants for the octree compute
        for (int i = 0; i < 2; ++i) {
            octant_usage_buffers[i] = webgpu_context->create_buffer(SDF_Globals.octants_max_size * sizeof(uint32_t), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "octant_usage");
            webgpu_context->update_buffer(octant_usage_buffers[i], 0, &zero_value, sizeof(uint32_t));
        }

        for (int i = 0; i < 4; ++i) {
            octant_usage_ping_pong_uniforms[i].data = octant_usage_buffers[i / 2];
            octant_usage_ping_pong_uniforms[i].binding = i % 2;
            octant_usage_ping_pong_uniforms[i].buffer_size = SDF_Globals.octants_max_size * sizeof(uint32_t);
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

void SculptManager::load_shaders() {
    evaluate_shader = RendererStorage::get_shader("data/shaders/octree/evaluator.wgsl");
    increment_level_shader = RendererStorage::get_shader("data/shaders/octree/increment_level.wgsl");
    write_to_texture_shader = RendererStorage::get_shader("data/shaders/octree/write_to_texture.wgsl");
    brick_removal_shader = RendererStorage::get_shader("data/shaders/octree/brick_removal.wgsl");
    brick_copy_shader = RendererStorage::get_shader("data/shaders/octree/brick_copy.wgsl");
    evaluation_initialization_shader = RendererStorage::get_shader("data/shaders/octree/initialization.wgsl");
    //brick_unmark_shader = RendererStorage::get_shader("data/shaders/octree/brick_unmark.wgsl");
    sculpt_delete_shader = RendererStorage::get_shader("data/shaders/octree/sculpture_delete.wgsl");
}

void SculptManager::init_pipelines_and_bindgroups() {
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    // Create bindgroups
    {
        std::vector<Uniform*> uniforms = { &merge_data_uniform, &octree_edit_list,
                                       &octree_stroke_context, &SDF_Globals.brick_buffers, &stroke_culling_data };
        evaluate_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 0);
    }

    // Brick removal pass
    {
        std::vector<Uniform*> uniforms = { &SDF_Globals.brick_buffers };
        indirect_brick_removal_bind_group = webgpu_context->create_bind_group(uniforms, brick_removal_shader, 0);
    }

    // Octree increment iteration pass
    {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &SDF_Globals.brick_buffers };
        increment_level_bind_group = webgpu_context->create_bind_group(uniforms, increment_level_shader, 0);
    }

    // Brick copy for brick instanced rendering
    {
        std::vector<Uniform*> uniforms = { &SDF_Globals.brick_copy_buffer, &SDF_Globals.brick_buffers };
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
        std::vector<Uniform*> uniforms = { &SDF_Globals.sdf_texture_uniform, &octree_edit_list, &stroke_culling_data,
                                           &octree_stroke_context, & SDF_Globals.brick_buffers, & SDF_Globals.sdf_material_texture_uniform };
        write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);
    }

    // Octree initialiation bindgroup
    {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                            &SDF_Globals.brick_buffers, &stroke_culling_data, &octree_stroke_context };

        evaluation_initialization_bind_group = webgpu_context->create_bind_group(uniforms, evaluation_initialization_shader, 0);
    }

    // Sculpt delete
    {
        Uniform alt_brick_uniform = SDF_Globals.brick_buffers;
        alt_brick_uniform.binding = 0u;

        std::vector<Uniform*> uniforms = { &alt_brick_uniform };
        sculpt_delete_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_delete_shader, 1u);
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
