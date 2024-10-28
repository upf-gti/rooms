#include "raymarching_renderer.h"

#include "engine/rooms_engine.h"
#include "rooms_renderer.h"

#include "framework/parsers/parse_scene.h"
#include "framework/math/intersections.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/camera/camera.h"
#include "framework/resources/sculpt.h"

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
}

int RaymarchingRenderer::initialize(bool use_mirror_screen)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    // Compute constants

#ifndef DISABLE_RAYMARCHER
    init_raymarching_proxy_pipeline();
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
    wgpuBindGroupRelease(render_camera_bind_group);
    wgpuBindGroupRelease(render_preview_camera_bind_group);
    wgpuBindGroupRelease(sculpt_data_bind_preview_group);

    delete render_proxy_shader;
    delete render_preview_proxy_shader;

    camera_uniform->destroy();
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

void RaymarchingRenderer::get_brick_usage(std::function<void(float, uint32_t)> callback)
{
    //WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    //sBrickBuffers_counters brick_usage_info;
    //webgpu_context->read_buffer(brick_buffers_counters_read_buffer, sizeof(sBrickBuffers_counters), &brick_usage_info);

    //uint32_t brick_count = brick_usage_info.brick_instance_counter;
    //float pct = brick_count / (float)max_brick_count;

    //callback(pct, brick_count);
}

void RaymarchingRenderer::render(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    render_raymarching_proxy(render_pass, camera_buffer_stride);

    if (render_preview) {
        render_preview_raymarching_proxy(render_pass, camera_buffer_stride);
    }
}

void RaymarchingRenderer::render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

#ifndef NDEBUG
    wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render sculpt proxy geometry");
#endif

    // Prepare the pipeline
    if (!render_proxy_geometry_pipeline.set(render_pass)) {
#ifndef NDEBUG
        wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif
        return;
    }

    // Set vertex buffer for teh cube mesh
    const Surface* surface = cube_mesh->get_surface(0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_vertices_byte_size());

    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_proxy_geometry_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, render_camera_bind_group, 1, &camera_buffer_stride);
    //wgpuRenderPassEncoderSetBindGroup(render_pass, 2, sculpt_data_bind_proxy_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 3, Renderer::instance->get_lighting_bind_group(), 0, nullptr);

    for (auto& it : rooms_renderer->get_sculpts_render_list()) {
        Sculpt* curr_sculpt = it.second->sculpt;

        if (curr_sculpt->is_deleted()) {
            continue;
        }

        const uint32_t curr_sculpt_instance_count = 0u;

        wgpuRenderPassEncoderSetBindGroup(render_pass, 2u, curr_sculpt->get_readonly_sculpt_bindgroup(), 0u, nullptr);

        // Instance stride
        //wgpuRenderPassEncoderSetBindGroup(render_pass, 4u, sculpt_instance_count_bindgroup, sculpt_buffer_instance_count, &offset);

        wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(curr_sculpt->get_indirect_render_buffer().data), 0u);
    }

#ifndef NDEBUG
    wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif
}

void RaymarchingRenderer::render_preview_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

#ifndef NDEBUG
    wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render preview proxy geometry");
#endif

    // Render Preview proxy geometry
    if (!render_preview_proxy_geometry_pipeline.set(render_pass)) {
#ifndef NDEBUG
        wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif
        return;
    }

    // Update sculpt data
    //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));

    const Surface* surface = cube_mesh->get_surface(0);

    uint8_t bind_group_index = 0;

    // Set bind groups
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_preview_camera_bind_group, 1, &camera_buffer_stride);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, sculpt_data_bind_preview_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 2, Renderer::instance->get_lighting_bind_group(), 0, nullptr);

    // Set vertex buffer while encoding the render pass
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_vertices_byte_size());

    // Submit indirect drawcalls
    wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 4u);

#ifndef NDEBUG
    wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif
}

void RaymarchingRenderer::init_raymarching_proxy_pipeline()
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();
    bool is_openxr_available = RoomsRenderer::instance->get_openxr_available();

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    cube_mesh = parse_mesh("data/meshes/cube.obj");

    render_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_plain.wgsl");

    {
        camera_uniform = rooms_renderer->get_current_camera_uniform();

        std::vector<Uniform*> uniforms = { &sdf_globals.linear_sampler_uniform, &rooms_renderer->get_global_sculpts_instance_data(),
            &sdf_globals.sdf_texture_uniform, & sdf_globals.brick_buffers,
            &sdf_globals.sdf_material_texture_uniform, &sdf_globals.preview_stroke_uniform_2 };

        render_proxy_geometry_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 0);
        uniforms = { camera_uniform };
        render_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_proxy_shader, 1);
    }

    WGPUTextureFormat swapchain_format = is_openxr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    RenderPipelineDescription desc = { .cull_mode = WGPUCullMode_Front };
    render_proxy_geometry_pipeline.create_render_async(render_proxy_shader, color_target, desc);

    // Proxy for Preview
    render_preview_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_preview.wgsl");
    {
        std::vector<Uniform*> uniforms;

        uniforms = { camera_uniform };
        render_preview_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 0);

        uniforms = {&sdf_globals.preview_stroke_uniform_2, &sdf_globals.brick_buffers, &rooms_renderer->get_global_sculpts_instance_data() };
        sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);
    }

    color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;
    render_preview_proxy_geometry_pipeline.create_render_async(render_preview_proxy_shader, color_target, desc);
}
