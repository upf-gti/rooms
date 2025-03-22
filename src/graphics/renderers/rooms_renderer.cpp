#include "rooms_renderer.h"

#include "graphics/shader.h"

#include "graphics/managers/sculpt_manager.h"

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/quad_mirror.wgsl.gen.h"

#if defined(OPENXR_SUPPORT)
#include "xr/openxr/openxr_context.h"
#elif defined(WEBXR_SUPPORT)
#include "xr/webxr/webxr_context.h"
#endif

#include "framework/nodes/environment_3d.h"
#include "engine/rooms_engine.h"

#include "graphics/renderer_storage.h"
#include "framework/camera/camera.h"
#include "framework/input.h"

void sSDFGlobals::clean()
{
    brick_buffers.destroy();
    brick_copy_aabb_gen_buffer.destroy();
    indirect_buffers.destroy();
    sdf_texture_uniform.destroy();
    sdf_material_texture_uniform.destroy();
    preview_stroke_uniform_2.destroy();
    preview_stroke_uniform.destroy();
    linear_sampler_uniform.destroy();
}

RoomsRenderer::RoomsRenderer(const sRendererConfiguration& config) : Renderer(config)
{

}

RoomsRenderer::~RoomsRenderer()
{
    if (sculpt_manager) {
        delete sculpt_manager;
    }

    Renderer::clean();
}

int RoomsRenderer::pre_initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::pre_initialize(window, use_mirror_screen);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    return 0;
}

int RoomsRenderer::initialize()
{
    return Renderer::initialize();
}

int RoomsRenderer::post_initialize()
{
    Renderer::post_initialize();

    init_sdf_globals();

    initialize_sculpt_render_instances();

    sculpt_manager = new SculptManager();
    sculpt_manager->init();

    raymarching_renderer.initialize();

    set_custom_pass_user_data(&raymarching_renderer);

    if (is_xr_available && use_mirror_screen && use_custom_mirror) {
        custom_mirror_texture = webgpu_context->create_texture(
            WGPUTextureDimension_2D,
            webgpu_context->xr_swapchain_format,
            { webgpu_context->screen_width, webgpu_context->screen_height, 1 },
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
            1,
            1
        );

        custom_mirror_texture_view = webgpu_context->create_texture_view(
            custom_mirror_texture,
            WGPUTextureViewDimension_2D,
            webgpu_context->xr_swapchain_format,
            WGPUTextureAspect_All,
            0,
            1,
            0,
            1,
            "custom_mirror_fbo"
        );

        custom_mirror_depth_texture = webgpu_context->create_texture(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_Depth32Float,
            { webgpu_context->screen_width, webgpu_context->screen_height, 1 },
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
            1,
            1
        );

        custom_mirror_depth_texture_view = webgpu_context->create_texture_view(
            custom_mirror_depth_texture,
            WGPUTextureViewDimension_2D,
            WGPUTextureFormat_Depth32Float,
            WGPUTextureAspect_All,
            0,
            1,
            0,
            1,
            "custom_mirror_fbo_depth"
        );

        custom_mirror_fbo_uniform.data = custom_mirror_texture_view;
        custom_mirror_fbo_uniform.binding = 0;

        std::vector<Uniform*> uniforms = { &custom_mirror_fbo_uniform, &linear_sampler_uniform };
        custom_mirror_fbo_bind_group = webgpu_context->create_bind_group(uniforms, mirror_shader, 0);

        set_custom_mirror_fbo_bind_group(custom_mirror_fbo_bind_group);

        custom_mirror_camera_uniform.data = webgpu_context->create_buffer(camera_buffer_stride, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "custom_mirror_camera_buffer");
        custom_mirror_camera_uniform.binding = 0;
        custom_mirror_camera_uniform.buffer_size = sizeof(sCameraData);

        uniforms = { &custom_mirror_camera_uniform };
        custom_mirror_camera_bind_group = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries), 1);
    }

#ifndef DISABLE_RAYMARCHER
    custom_post_opaque_pass = [](WGPURenderPassEncoder render_pass, WGPUBindGroup camera_bind_group, void* user_data, uint32_t camera_stride_offset = 0) {
        RaymarchingRenderer* raymarching_renderer = reinterpret_cast<RaymarchingRenderer*>(user_data);
        raymarching_renderer->render(render_pass, camera_bind_group, camera_stride_offset);
    };
#endif

    return 0;
}

void RoomsRenderer::clean()
{
    sculpt_instances.uniform_count_buffer.destroy();
    wgpuBindGroupRelease(sculpt_instances.count_bindgroup);
    delete sculpt_instances.prepare_indirect_shader;

    sdf_globals.clean();

    sculpt_manager->clean();

    raymarching_renderer.clean();

    global_sculpts_instance_data_uniform.destroy();


    if (is_xr_available && use_mirror_screen && use_custom_mirror) {
        wgpuTextureRelease(custom_mirror_texture);
        wgpuTextureViewRelease(custom_mirror_texture_view);

        wgpuTextureRelease(custom_mirror_depth_texture);
        wgpuTextureViewRelease(custom_mirror_depth_texture_view);

        custom_mirror_fbo_uniform.destroy();

        wgpuBindGroupRelease(custom_mirror_fbo_bind_group);
        custom_mirror_camera_uniform.destroy();

        wgpuBindGroupRelease(custom_mirror_camera_bind_group);

        for (int i = 0; i < RENDER_LIST_COUNT; ++i) {
            custom_mirror_instances_data.instances_data_uniforms[i].destroy();

            if (custom_mirror_instances_data.instances_bind_groups[i]) {
                wgpuBindGroupRelease(custom_mirror_instances_data.instances_bind_groups[i]);
            }
        }
    }
}

void RoomsRenderer::init_sdf_globals()
{
    // Size of penultimate level
    sdf_globals.octants_max_size = pow(floorf(SDF_RESOLUTION / static_cast<float>(ATLAS_BRICK_SIZE)), 3.0f);

    sdf_globals.octree_depth = static_cast<uint8_t>(OCTREE_DEPTH);

    // total size considering leaves and intermediate levels
    sdf_globals.octree_total_size = (pow(8, sdf_globals.octree_depth + 1) - 1) / 7;

    sdf_globals.octree_last_level_size = sdf_globals.octree_total_size - (pow(8, sdf_globals.octree_depth) - 1) / 7;

    uint32_t brick_count_in_axis = static_cast<uint32_t>(SDF_RESOLUTION / ATLAS_BRICK_SIZE);
    sdf_globals.max_brick_count = brick_count_in_axis * brick_count_in_axis * brick_count_in_axis;

    sdf_globals.empty_brick_and_removal_buffer_count = sdf_globals.max_brick_count + (sdf_globals.max_brick_count % 4);

    sdf_globals.workgroup_brick_process_count = (uint32_t)glm::ceil(sdf_globals.octree_last_level_size / 512.0);

    // number of bricks that fit in one axis
    float num_bricks_in_octree_axis = powf(2.0, sdf_globals.octree_depth);

    //uint32_t p = sdf_globals.octree_total_size - sdf_globals.octree_last_level_size;

    Shader::set_custom_define("NUM_BRICKS_IN_OCTREE_AXIS", num_bricks_in_octree_axis);
    Shader::set_custom_define("ATLAS_BRICK_SIZE", static_cast<float>(ATLAS_BRICK_SIZE));
    Shader::set_custom_define("ATLAS_BRICK_NO_BORDER_SIZE", static_cast<float>(ATLAS_BRICK_NO_BORDER_SIZE));
    Shader::set_custom_define("OCTREE_DEPTH", sdf_globals.octree_depth);
    Shader::set_custom_define("OCTREE_TOTAL_SIZE", sdf_globals.octree_total_size);
    Shader::set_custom_define("PREVIEW_PROXY_BRICKS_COUNT", PREVIEW_PROXY_BRICKS_COUNT);
    Shader::set_custom_define("BRICK_REMOVAL_COUNT", sdf_globals.empty_brick_and_removal_buffer_count);
    Shader::set_custom_define("MAX_SUBDIVISION_SIZE", sdf_globals.octree_last_level_size);
    Shader::set_custom_define("MAX_STROKE_INFLUENCE_COUNT", MAX_STROKE_INFLUENCE_COUNT);
    Shader::set_custom_define("OCTREE_LAST_LEVEL_STARTING_IDX", sdf_globals.octree_total_size - sdf_globals.octree_last_level_size);

    //Shader::set_custom_define("WORKGROUP_BRICK_COUNT", sdf_globals.workgroup_brick_process_count);

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);

    sdf_globals.brick_world_size = (SCULPT_MAX_SIZE / num_bricks_in_octree_axis);

    {
        // Atlas texture
        sdf_globals.sdf_texture.create(
            WGPUTextureDimension_3D,
            WGPUTextureFormat_R32Float,
            { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
            static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
            1, 1, nullptr);

        sdf_globals.sdf_texture_uniform.data = sdf_globals.sdf_texture.get_view(WGPUTextureViewDimension_3D);
        sdf_globals.sdf_texture_uniform.binding = 3;
    }

    {
        // Material atlas texture
        sdf_globals.sdf_material_texture.create(
            WGPUTextureDimension_3D,
            WGPUTextureFormat_R32Uint,
            { SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION },
            static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc),
            1, 1, nullptr);

        sdf_globals.sdf_material_texture_uniform.data = sdf_globals.sdf_material_texture.get_view(WGPUTextureViewDimension_3D);
        sdf_globals.sdf_material_texture_uniform.binding = 8; // TODO: set as 4
    }

    {
        // Create the brick Buffers
        // An struct that contines: a empty brick counter in the atlas, the empty brick buffer, and the data off all the instances
        uint32_t default_val = 0u;
        uint32_t struct_size =
            sizeof(uint32_t) * 4u + sizeof(uint32_t) * sdf_globals.max_brick_count                          // Atlas empty buffer counter, padding & index buffer
            + sizeof(uint32_t) * 4u + sizeof(uint32_t) * sdf_globals.empty_brick_and_removal_buffer_count   // brick removal counter, padding & index buffer
            + sizeof(uint32_t) * 4u + sdf_globals.max_brick_count * sizeof(ProxyInstanceData)               // Brick counter, padding & instance buffer
            + sizeof(uint32_t) * 4u + PREVIEW_PROXY_BRICKS_COUNT * sizeof(ProxyInstanceData);   // Preview brick counter, padding & instance buffer
        std::vector<uint8_t> default_bytes(struct_size, 0);
        sdf_globals.brick_buffers.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage, default_bytes.data(), "brick_buffer_struct");
        sdf_globals.brick_buffers.binding = 5;
        sdf_globals.brick_buffers.buffer_size = struct_size;

    }

    //brick_buffers_counters_read_buffer = webgpu_context->create_buffer(sizeof(sBrickBuffers_counters), WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, nullptr, "brick counters read buffer");

    {
        // Empty atlas indices data
        uint32_t* atlas_indices = new uint32_t[sdf_globals.octants_max_size + 4u];
        atlas_indices[0] = sdf_globals.octants_max_size;
        atlas_indices[1] = 0u;
        atlas_indices[2] = 0u;
        atlas_indices[3] = 0u;

        for (uint32_t i = 0u; i < sdf_globals.octants_max_size; i++) {
            atlas_indices[i + 4u] = sdf_globals.octants_max_size - i - 1u;
        }

        webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.brick_buffers.data), 0u, atlas_indices, sizeof(uint32_t)* (sdf_globals.octants_max_size + 4u));
        delete[] atlas_indices;
    }

    {
        // Preview
        uint32_t struct_size = sizeof(sGPUStroke) + sizeof(Edit) * PREVIEW_BASE_EDIT_LIST;
        sdf_globals.preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_buffer");
        sdf_globals.preview_stroke_uniform.binding = 0;
        sdf_globals.preview_stroke_uniform.buffer_size = struct_size;

        sdf_globals.preview_stroke_uniform_2.data = sdf_globals.preview_stroke_uniform.data;
        sdf_globals.preview_stroke_uniform_2.binding = 1u;
        sdf_globals.preview_stroke_uniform_2.buffer_size = sdf_globals.preview_stroke_uniform.buffer_size;
    }

    {
        sdf_globals.linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
        sdf_globals.linear_sampler_uniform.binding = 4;
    }


    // Indirect dispatch buffer
    {
        uint32_t buffer_size = sizeof(uint32_t) * 4u * 4u;
        sdf_globals.indirect_buffers.data = webgpu_context->create_buffer(buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Indirect | WGPUBufferUsage_Storage, nullptr, "indirect_buffers_struct");
        sdf_globals.indirect_buffers.binding = 8u;
        sdf_globals.indirect_buffers.buffer_size = buffer_size;

        uint32_t default_indirect_values[16u] = {
            36u, 0u, 0u, 0u, // bricks indirect call
            36u, 0u, 0u, 0u,// preview bricks indirect call
            0u, 1u, 1u, 0u, // brick removal call (1 padding)
            1u, 1u, 1u, 0u // octree subdivision
        };
        webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.indirect_buffers.data), 0, default_indirect_values, sizeof(uint32_t) * 16u);
    }
}

void RoomsRenderer::initialize_sculpt_render_instances()
{
    // Sculpt model buffer
    {
        // TODO(Juan): make this array dinamic
        uint32_t size = sizeof(sSculptInstanceData) * 512u; // The current max size of instances
        global_sculpts_instance_data_uniform.data = webgpu_context->create_buffer(size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 0u, "sculpt instance data");
        global_sculpts_instance_data_uniform.binding = 9u;
        global_sculpts_instance_data_uniform.buffer_size = size;
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

    sculpt_instances.prepare_indirect.create_compute_async(sculpt_instances.prepare_indirect_shader);
}

void RoomsRenderer::update(float delta_time)
{
    Renderer::update(delta_time);

    sculpt_manager->update(global_command_encoder);

    upload_sculpt_models_and_instances(global_command_encoder);

    update_sculpts_indirect_buffers(global_command_encoder);
}

void RoomsRenderer::render()
{
    if (is_xr_available && use_mirror_screen && use_custom_mirror) {

        RoomsEngine* rooms_engine = static_cast<RoomsEngine*>(RoomsEngine::get_instance());
        Environment3D* environment = rooms_engine->get_environment();

        environment->set_position(camera_3d->get_eye());

        render_custom_mirror_fbo();

        environment->set_position(xr_context->per_view_data[0].position);
    }

    Renderer::render();

#ifndef __EMSCRIPTEN__
    if (!last_frame_timestamps.empty()) {
        std::map<uint8_t, std::string>& queries_map = get_queries_label_map();
        for (int i = 0; i < last_frame_timestamps.size(); ++i) {
            std::string label = queries_map[i * 2 + 1];

            if (label == "evaluation") {
                last_evaluation_time = last_frame_timestamps[i];
            }
        }
    }
#endif

    if (get_sculpt_manager()->has_performed_evaluation()) {
        sculpt_manager->read_GPU_results();
    }

    // For the next frame
    sculpts_render_lists.clear();
}

void RoomsRenderer::render_custom_mirror_fbo()
{
    camera_data.exposure = exposure;
    camera_data.ibl_intensity = ibl_intensity;
    camera_data.screen_size = { webgpu_context->screen_width, webgpu_context->screen_height };

    std::vector<std::vector<sRenderData>> render_lists(RENDER_LIST_COUNT);

    prepare_cull_instancing(*camera_3d, render_lists, custom_mirror_instances_data);

    // Update main 3d camera

    camera_data.eye = camera_3d->get_eye();
    camera_data.view_projection = camera_3d->get_view_projection();
    camera_data.view = camera_3d->get_view();
    camera_data.projection = camera_3d->get_projection();

    wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(custom_mirror_camera_uniform.data), 0, &camera_data, sizeof(sCameraData));

    render_camera(render_lists, custom_mirror_texture_view, custom_mirror_depth_texture_view, custom_mirror_instances_data, custom_mirror_camera_bind_group, true, "custom_mirror_pass", 0);
}

void RoomsRenderer::update_sculpts_indirect_buffers(WGPUCommandEncoder command_encoder)
{
    /*if (!sculpt_manager->has_performed_evaluation()) {
        return;
    }*/

    WGPUComputePassDescriptor compute_pass_desc = { .label = { "update_sculpts_indirect_buffers_pass", WGPU_STRLEN} };
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(global_command_encoder, &compute_pass_desc);

#ifndef NDEBUG
    webgpu_context->push_debug_group(compute_pass, { "Update the sculpts instances", WGPU_STRLEN });
#endif

    if (sculpts_render_lists.size() > 0u) {
        sculpt_instances.prepare_indirect.set(compute_pass);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0u, sculpt_instances.count_bindgroup, 0u, nullptr);

        for (auto& it : sculpts_render_lists) {
            Sculpt* current_sculpt = it.second.sculpt;
            wgpuComputePassEncoderSetBindGroup(compute_pass, 1u, current_sculpt->get_sculpt_bindgroup(), 0u, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1u, 1u, 1u);
        }
    }

#ifndef NDEBUG
    webgpu_context->pop_debug_group(compute_pass);
#endif
    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
}

void RoomsRenderer::upload_sculpt_models_and_instances(WGPUCommandEncoder command_encoder)
{
    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    WebGPUContext* webgpu_context = rooms_renderer->get_webgpu_context();
    sSDFGlobals& sdf_globals = rooms_renderer->get_sdf_globals();

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

        it.second.sculpt->set_in_frame_model_buffer_index(in_frame_instance_count);

        for (uint16_t i = 0u; i < it.second.instance_count; i++) {
            models_for_upload[in_frame_instance_count++] = (it.second.models[i]);
        }

        sculpt_instances.count_buffer[sculpt_to_render_count++] = it.second.instance_count;
    }

    // Upload the count of instances per sculpt
    webgpu_context->update_buffer(std::get<WGPUBuffer>(sculpt_instances.uniform_count_buffer.data), 0u, sculpt_instances.count_buffer.data(), sizeof(uint32_t) * sculpt_to_render_count);

    // Upload all the instance data
    webgpu_context->update_buffer(std::get<WGPUBuffer>(global_sculpts_instance_data_uniform.data), 0u, models_for_upload.data(), sizeof(sSculptInstanceData) * in_frame_instance_count);
}

uint32_t RoomsRenderer::add_sculpt_render_call(Sculpt* sculpt, const glm::mat4& model, const uint32_t flags)
{
    const uint32_t sculpt_id = sculpt->get_sculpt_id();

    sSculptRenderInstances* render_instance = nullptr;

    if (!sculpts_render_lists.contains(sculpt_id)) {
        sculpts_render_lists[sculpt_id] = sSculptRenderInstances{ .sculpt = sculpt, .instance_count = 0u };
    }
    
    render_instance = &sculpts_render_lists[sculpt_id];

    assert(render_instance->instance_count < MAX_INSTANCES_PER_SCULPT && "MAX NUM OF SCULPT INSTANCES");

    render_instance->models[render_instance->instance_count] = {
        .flags = flags,
        .instance_id = render_instance->instance_count,
        .model = model,
        .inv_model = glm::inverse(model)
    };

    return render_instance->instance_count++;
}

bool RoomsRenderer::has_performed_evaluation() const
{
    return sculpt_manager->has_performed_evaluation();
}
