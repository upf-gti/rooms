#include "raymarching_renderer.h"

#include "engine/rooms_engine.h"
#include "rooms_renderer.h"

#include "framework/parsers/parse_scene.h"
#include "framework/math/intersections.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/input.h"
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
    
#ifndef DISABLE_RAYMARCHER
    init_octree_ray_intersection_pipeline();
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

    delete render_proxy_shader;
    delete render_preview_proxy_shader;
#endif
}

void RaymarchingRenderer::add_rendercall_to_sculpt(Sculpt* sculpt, const glm::mat4& model, const uint32_t flags)
{
    const uint32_t sculpt_id = sculpt->get_sculpt_id();

    sSculptRenderInstances* render_instance = nullptr;

    if (!sculpts_render_lists.contains(sculpt_id)) {
        render_instance = new sSculptRenderInstances{.sculpt = sculpt, .instance_count = 0u};

        sculpts_render_lists[sculpt_id] = render_instance;
    } else {
        render_instance = sculpts_render_lists[sculpt_id];
    }

    assert(render_instance->instance_count < MAX_INSTANCES_PER_SCULPT && "MAX NUM OF SCULPT INSTANCES");

    render_instance->models[render_instance->instance_count++] = { .flags = flags, .model = model };
}

void RaymarchingRenderer::update_sculpts_and_instances(WGPUCommandEncoder command_encoder)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

    WGPUComputePassDescriptor compute_pass_desc = {};
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

#ifndef NDEBUG
    wgpuComputePassEncoderPushDebugGroup(compute_pass, "Update the sculpts instances");
#endif

    // Prepare the instances of the sculpts that are rendered on the curren frame
    // Generate buffer of instances
    uint32_t in_frame_instance_count = 0u;
    uint32_t sculpt_to_render_count = 2u;
    sculpt_instances.count_buffer[0u] = 0u;
    sculpt_instances.count_buffer[1u] = 0u;
    for (auto& it : sculpts_render_lists) {
        if (models_for_upload.capacity() <= in_frame_instance_count) {
            models_for_upload.resize(models_for_upload.capacity() + 10u);
        }

        for (uint16_t i = 0u; i < it.second->instance_count; i++) {
            models_for_upload[in_frame_instance_count++] = (it.second->models[i]);
        }

        sculpt_instances.count_buffer[sculpt_to_render_count++] = it.second->instance_count;
    }

    // Upload the count of instances per sculpt
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_instances.uniform_count_buffer.data), 0u, sculpt_instances.count_buffer.data(), sizeof(uint32_t) * sculpt_to_render_count);

    // Upload all the instance data
    webgpu_context->update_buffer(std::get<WGPUBuffer>(global_sculpts_instance_data_uniform.data), 0u, models_for_upload.data(), sizeof(sSculptInstanceData) * in_frame_instance_count);

    uint32_t offset = 0u;
    for (auto& it : sculpts_render_lists) {
        Sculpt* current_sculpt = it.second->sculpt;

        sculpt_instances.prepare_indirect.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, sculpt_instances.count_bindgroup, 0u, nullptr);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, current_sculpt->get_sculpt_bindgroup(), 0u, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);

        offset++;
    }

#ifndef NDEBUG
    wgpuComputePassEncoderPopDebugGroup(compute_pass);
#endif

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
#ifndef DISABLE_RAYMARCHER
    if (Input::is_mouse_pressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
       /* RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
        WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

        Camera* camera = rooms_renderer->get_camera();
        glm::vec3 ray_dir = camera->screen_to_ray(Input::get_mouse_position());

        octree_ray_intersect(camera->get_eye(), glm::normalize(ray_dir));*/
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

void RaymarchingRenderer::octree_ray_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, std::function<void(glm::vec3)> callback)
{
    WebGPUContext* webgpu_context = RoomsRenderer::instance->get_webgpu_context();

    RoomsEngine* engine_instance = static_cast<RoomsEngine*>(RoomsEngine::instance);
    SculptEditor* sculpt_editor = engine_instance->get_sculpt_editor();
    //SculptInstance* current_sculpt = sculpt_editor->get_current_sculpt();

    //if (!current_sculpt) {
    //    spdlog::warn("Can not ray intersect without current sculpt");
    //    return;
    //}

    //// Initialize a command encoder
    //WGPUCommandEncoderDescriptor encoder_desc = {};
    //WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    //// Create compute_raymarching pass
    //WGPUComputePassDescriptor compute_pass_desc = {};
    //compute_pass_desc.timestampWrites = nullptr;
    //WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_desc);

    //// Convert ray, origin from world to sculpt space

    //glm::quat inv_rotation = glm::inverse(current_sculpt->get_rotation());

    //ray_info.ray_origin = ray_origin - current_sculpt->get_translation();
    //ray_info.ray_origin = inv_rotation * ray_info.ray_origin;

    //ray_info.ray_dir = ray_dir;
    //ray_info.ray_dir = inv_rotation * ray_info.ray_dir;

    //webgpu_context->update_buffer(std::get<WGPUBuffer>(ray_info_uniform.data), 0, &ray_info, sizeof(RayInfo));

    //compute_octree_ray_intersection_pipeline.set(compute_pass);

    //wgpuComputePassEncoderSetBindGroup(compute_pass, 0, octree_ray_intersection_bind_group, 0, nullptr);
    //wgpuComputePassEncoderSetBindGroup(compute_pass, 1, octree_ray_intersection_info_bind_group, 0, nullptr);
    //wgpuComputePassEncoderSetBindGroup(compute_pass, 2, sculpt_octree_bindgroup, 0, nullptr);

    //wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);

    //// Finalize compute_raymarching pass
    //wgpuComputePassEncoderEnd(compute_pass);

    //wgpuCommandEncoderCopyBufferToBuffer(command_encoder, std::get<WGPUBuffer>(ray_intersection_info_uniform.data), 0, ray_intersection_info_read_buffer, 0, sizeof(RayIntersectionInfo));

    //WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    //cmd_buff_descriptor.nextInChain = NULL;
    //cmd_buff_descriptor.label = "Ray Intersection Command Buffer";

    //// Encode and submit the GPU commands
    //WGPUCommandBuffer commands = wgpuCommandEncoderFinish(command_encoder, &cmd_buff_descriptor);
    //wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    //wgpuCommandBufferRelease(commands);
    //wgpuComputePassEncoderRelease(compute_pass);
    //wgpuCommandEncoderRelease(command_encoder);

    //webgpu_context->read_buffer(ray_intersection_info_read_buffer, sizeof(RayIntersectionInfo), &ray_intersection_info);
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

void RaymarchingRenderer::render_raymarching_proxy(WGPURenderPassEncoder render_pass, uint32_t camera_buffer_stride)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

    
#ifndef NDEBUG
    wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render sculpt proxy geometry");
#endif

    // Prepare the pipeline
    render_proxy_geometry_pipeline.set(render_pass);
    // Set vertex buffer for teh cube mesh
    const Surface* surface = cube_mesh->get_surface(0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_vertices_byte_size());

    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_proxy_geometry_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, render_camera_bind_group, 1, &camera_buffer_stride);
    //wgpuRenderPassEncoderSetBindGroup(render_pass, 2, sculpt_data_bind_proxy_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 3, Renderer::instance->get_lighting_bind_group(), 0, nullptr);

    for (auto& it : sculpts_render_lists) {
        Sculpt* curr_sculpt = it.second->sculpt;
        const uint32_t curr_sculpt_instance_count = 0u;

        wgpuRenderPassEncoderSetBindGroup(render_pass, 2u, curr_sculpt->get_readonly_sculpt_bindgroup(), 0u, nullptr);

        // Instance stride
        //wgpuRenderPassEncoderSetBindGroup(render_pass, 4u, sculpt_instance_count_bindgroup, sculpt_buffer_instance_count, &offset);

        wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(curr_sculpt->get_indirect_render_buffer().data), 0u);
    }

    sculpts_render_lists.clear();

#ifndef NDEBUG
    wgpuRenderPassEncoderPopDebugGroup(render_pass);
#endif

#ifndef NDEBUG
        wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render sculpt proxy geometry");
#endif

        {
//#ifndef NDEBUG
//    wgpuRenderPassEncoderPopDebugGroup(render_pass);
//    wgpuRenderPassEncoderPushDebugGroup(render_pass, "Render preview proxy geometry");
//#endif
//    // Render Preview proxy geometry
//    if (static_cast<RoomsEngine*>(RoomsEngine::instance)->get_current_editor_type() == EditorType::SCULPT_EDITOR) {
//        render_preview_proxy_geometry_pipeline.set(render_pass);
//
//        // Update sculpt data
//        //webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_data_uniform.data), 0, &sculpt_data, sizeof(sSculptData));
//
//        const Surface* surface = cube_mesh->get_surface(0);
//
//        uint8_t bind_group_index = 0;
//
//        // Set bind groups
//        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, render_preview_camera_bind_group, 1, &camera_buffer_stride);
//        wgpuRenderPassEncoderSetBindGroup(render_pass, 1, sculpt_data_bind_preview_group, 0, nullptr);
//        wgpuRenderPassEncoderSetBindGroup(render_pass, 2, Renderer::instance->get_lighting_bind_group(), 0, nullptr);
//
//        // Set vertex buffer while encoding the render pass
//        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, surface->get_vertex_buffer(), 0, surface->get_vertices_byte_size());
//
//        // Submit indirect drawcalls
//        wgpuRenderPassEncoderDrawIndirect(render_pass, std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), sizeof(uint32_t) * 4u);
//    }
//
        }
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

    // Sculpt model buffer
    {
        // TODO(Juan): make this array dinamic
        uint32_t size = sizeof(sSculptInstanceData) * 512u; // The current max size of instances
        global_sculpts_instance_data_uniform.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 0u, "sculpt instance data");
        global_sculpts_instance_data_uniform.binding = 9u;
        global_sculpts_instance_data_uniform.buffer_size = size;
    }

    {
        linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
        linear_sampler_uniform.binding = 4;
    }

    sculpt_instances.prepare_indirect_shader = RendererStorage::get_shader("data/shaders/octree/prepare_indirect_sculpt_render.wgsl");

    {
        uint32_t size = sizeof(uint32_t) * 22u;
        sculpt_instances.count_buffer.resize(22u);
        memset(sculpt_instances.count_buffer.data(), 1u, size);
        sculpt_instances.uniform_count_buffer.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sculpt_instances.count_buffer.data(), "sculpt index data");
        sculpt_instances.uniform_count_buffer.binding = 0u;
        sculpt_instances.uniform_count_buffer.buffer_size = size;

        std::vector<Uniform*> uniforms = { &sculpt_instances.uniform_count_buffer };
        sculpt_instances.count_bindgroup = webgpu_context->create_bind_group(uniforms, sculpt_instances.prepare_indirect_shader, 0);
    }

    {
        camera_uniform = rooms_renderer->get_current_camera_uniform();

        std::vector<Uniform*> uniforms = { &linear_sampler_uniform, &global_sculpts_instance_data_uniform,
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

    PipelineDescription desc = { .cull_mode = WGPUCullMode_Front };
    render_proxy_geometry_pipeline.create_render_async(render_proxy_shader, color_target, desc);

    // Proxy for Preview
    render_preview_proxy_shader = RendererStorage::get_shader("data/shaders/octree/proxy_geometry_preview.wgsl");
    {
        std::vector<Uniform*> uniforms;

        uniforms = { camera_uniform };
        render_preview_camera_bind_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 0);

        uniforms = {&sdf_globals.preview_stroke_uniform_2, &sdf_globals.brick_buffers, &global_sculpts_instance_data_uniform };
        sculpt_data_bind_preview_group = webgpu_context->create_bind_group(uniforms, render_preview_proxy_shader, 1);
    }

    color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;
    render_preview_proxy_geometry_pipeline.create_render_async(render_preview_proxy_shader, color_target, desc);
    sculpt_instances.prepare_indirect.create_compute_async(sculpt_instances.prepare_indirect_shader);
}

void RaymarchingRenderer::init_octree_ray_intersection_pipeline()
{
    //compute_octree_ray_intersection_shader = RendererStorage::get_shader("data/shaders/octree/octree_ray_intersection.wgsl");

    //RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    //sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();
    //WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();

    //// Ray Octree intersection bindgroup
    //{
    //    linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
    //    linear_sampler_uniform.binding = 4;

    //    std::vector<Uniform*> uniforms = { &sdf_globals.sdf_texture_uniform, &linear_sampler_uniform, &sdf_globals.sdf_material_texture_uniform };
    //    octree_ray_intersection_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 0);
    //}

    //// Ray Octree intersection info bindgroup
    //{
    //    ray_info_uniform.data = webgpu_context->create_buffer(sizeof(RayInfo), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "ray info");
    //    ray_info_uniform.binding = 0;
    //    ray_info_uniform.buffer_size = sizeof(RayInfo);

    //    ray_intersection_info_uniform.data = webgpu_context->create_buffer(sizeof(RayIntersectionInfo), WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage, nullptr, "ray intersection info");
    //    ray_intersection_info_uniform.binding = 3;
    //    ray_intersection_info_uniform.buffer_size = sizeof(RayIntersectionInfo);

    //    ray_intersection_info_read_buffer = webgpu_context->create_buffer(sizeof(RayIntersectionInfo), WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, nullptr, "ray intersection info read buffer");

    //    std::vector<Uniform*> uniforms = { &ray_info_uniform, &ray_intersection_info_uniform };
    //    octree_ray_intersection_info_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 1);
    //}

    ////{
    ////    std::vector<Uniform*> uniforms = { &sculpt_data_uniform };
    ////    sculpt_data_ray_bind_group = webgpu_context->create_bind_group(uniforms, compute_octree_ray_intersection_shader, 2);
    ////}

    //compute_octree_ray_intersection_pipeline.create_compute_async(compute_octree_ray_intersection_shader);
}
