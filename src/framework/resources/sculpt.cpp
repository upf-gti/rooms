#include "sculpt.h"

#include "graphics/shader.h"
#include "graphics/renderers/rooms_renderer.h"

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
    webgpu_context->update_buffer(std::get<WGPUBuffer>(SDF_Globals.octree_brick_buffers.data), sizeof(uint32_t), &(zero), sizeof(uint32_t));

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
            compute_octree_initialization_pipeline.set(compute_pass);

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_octree_initialization_bind_group, 0, nullptr);
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
                wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_bind_groups[ping_pong_idx], 0, nullptr);
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
            wgpuComputePassEncoderSetBindGroup(compute_pass, 2, octant_usage_bind_groups[ping_pong_idx], 0, nullptr);

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
                                           &octree_stroke_context, & SDF_Globals.octree_brick_buffers, & SDF_Globals.sdf_material_texture_uniform };

        wgpuBindGroupRelease(write_to_texture_bind_group);
        write_to_texture_bind_group = webgpu_context->create_bind_group(uniforms, write_to_texture_shader, 0);


        uniforms = { &merge_data_uniform, &octree_edit_list,
                     &octree_stroke_context, & SDF_Globals.octree_brick_buffers, &stroke_culling_data };

        wgpuBindGroupRelease(evaluate_bind_group);
        evaluate_bind_group = webgpu_context->create_bind_group(uniforms, evaluate_shader, 0);
    }

    if (recreated_stroke_context_buffer) {
        std::vector<Uniform*> uniforms = { &octree_indirect_buffer_struct, &octant_usage_initialization_uniform[0], &octant_usage_initialization_uniform[1],
                                        &SDF_Globals.octree_brick_buffers, &stroke_culling_data, &octree_stroke_context };

        compute_octree_initialization_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_initialization_shader, 0);
    }

    // Upload the data to the GPU
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_edit_list.data), 0, edits_to_upload.data(), sizeof(Edit) * stroke_manager.edit_list_count);
    webgpu_context->update_buffer(std::get<WGPUBuffer>(octree_stroke_context.data), sizeof(uint32_t) * 4 * 4, strokes_to_compute.data(), stroke_to_compute->in_frame_influence.stroke_count * sizeof(sToUploadStroke));

    //spdlog::info("min aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_min.x, stroke_to_compute->in_frame_influence.eval_aabb_min.y, stroke_to_compute->in_frame_influence.eval_aabb_min.z);
    //spdlog::info("max aabb: {}, {}, {}", stroke_to_compute->in_frame_influence.eval_aabb_max.x, stroke_to_compute->in_frame_influence.eval_aabb_max.y, stroke_to_compute->in_frame_influence.eval_aabb_max.z);
}
