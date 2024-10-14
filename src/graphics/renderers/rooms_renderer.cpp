#include "rooms_renderer.h"

#include "graphics/shader.h"

#include "graphics/managers/sculpt_manager.h"

RoomsRenderer::RoomsRenderer() : Renderer()
{

}

RoomsRenderer::~RoomsRenderer()
{

}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    init_sdf_globals();

    sculpt_manager = new SculptManager();
    sculpt_manager->init();

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    raymarching_renderer.initialize(use_mirror_screen);

    set_custom_pass_user_data(&raymarching_renderer);

#ifndef DISABLE_RAYMARCHER
    custom_post_opaque_pass = [](void* user_data, WGPURenderPassEncoder render_pass, uint32_t camera_stride_offset = 0) {
        RaymarchingRenderer* raymarching_renderer = reinterpret_cast<RaymarchingRenderer*>(user_data);
        raymarching_renderer->render(render_pass, camera_stride_offset);
    };
#endif

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    sculpt_manager->clean();
    raymarching_renderer.clean();

    if (sculpt_manager) {
        delete sculpt_manager;
    }
}

void RoomsRenderer::init_sdf_globals()
{
    // Size of penultimate level
    sdf_globals.octants_max_size = pow(floorf(SDF_RESOLUTION / 10.0f), 3.0f);

    sdf_globals.octree_depth = static_cast<uint8_t>(OCTREE_DEPTH);

    // total size considering leaves and intermediate levels
    sdf_globals.octree_total_size = (pow(8, sdf_globals.octree_depth + 1) - 1) / 7;

    sdf_globals.octree_last_level_size = sdf_globals.octree_total_size - (pow(8, sdf_globals.octree_depth) - 1) / 7;

    uint32_t brick_count_in_axis = static_cast<uint32_t>(SDF_RESOLUTION / BRICK_SIZE);
    sdf_globals.max_brick_count = brick_count_in_axis * brick_count_in_axis * brick_count_in_axis;

    sdf_globals.empty_brick_and_removal_buffer_count = sdf_globals.max_brick_count + (sdf_globals.max_brick_count % 4);
    float octree_space_scale = powf(2.0, sdf_globals.octree_depth + 3);

    uint32_t p = sdf_globals.octree_total_size - sdf_globals.octree_last_level_size;

    // Scale the size of a brick
    Shader::set_custom_define("WORLD_SPACE_SCALE", octree_space_scale); // Worldspace scale is 1/octree_max_width
    Shader::set_custom_define("OCTREE_DEPTH", sdf_globals.octree_depth);
    Shader::set_custom_define("OCTREE_TOTAL_SIZE", sdf_globals.octree_total_size);
    Shader::set_custom_define("PREVIEW_PROXY_BRICKS_COUNT", PREVIEW_PROXY_BRICKS_COUNT);
    Shader::set_custom_define("BRICK_REMOVAL_COUNT", sdf_globals.empty_brick_and_removal_buffer_count);
    Shader::set_custom_define("MAX_SUBDIVISION_SIZE", sdf_globals.octree_last_level_size);
    Shader::set_custom_define("MAX_STROKE_INFLUENCE_COUNT", MAX_STROKE_INFLUENCE_COUNT);
    Shader::set_custom_define("OCTREE_LAST_LEVEL_STARTING_IDX", sdf_globals.octree_total_size - sdf_globals.octree_last_level_size);

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);

    sdf_globals.brick_world_size = (SCULPT_MAX_SIZE / octree_space_scale) * 8.0f;

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

        webgpu_context->update_buffer(std::get<WGPUBuffer>(sdf_globals.brick_buffers.data), 0u, atlas_indices, sizeof(uint32_t) * (sdf_globals.octants_max_size + 4u));
        delete[] atlas_indices;
    }

    {
        // Preview
        uint32_t struct_size = sizeof(sToUploadStroke) + sizeof(Edit) * PREVIEW_BASE_EDIT_LIST;
        sdf_globals.preview_stroke_uniform.data = webgpu_context->create_buffer(struct_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "preview_stroke_buffer");
        sdf_globals.preview_stroke_uniform.binding = 0;
        sdf_globals.preview_stroke_uniform.buffer_size = struct_size;

        sdf_globals.preview_stroke_uniform_2.data = sdf_globals.preview_stroke_uniform.data;
        sdf_globals.preview_stroke_uniform_2.binding = 1u;
        sdf_globals.preview_stroke_uniform_2.buffer_size = sdf_globals.preview_stroke_uniform.buffer_size;
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

    //{
    //    // Indirect buffer for octree generation compute
    //    sdf_globals.brick_copy_buffer.data = webgpu_context->create_buffer(sizeof(uint32_t) * sdf_globals.octants_max_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, nullptr, "brick_copy_buffer");
    //    sdf_globals.brick_copy_buffer.binding = 0;
    //    sdf_globals.brick_copy_buffer.buffer_size = sizeof(uint32_t) * sdf_globals.octants_max_size;
    //}
}

void RoomsRenderer::update(float delta_time)
{
    Renderer::update(delta_time);

    sculpt_manager->update(global_command_encoder);

    raymarching_renderer.update_sculpts_and_instances(global_command_encoder);
}

void RoomsRenderer::render()
{
    Renderer::render();

    last_frame_timestamps = get_timestamps();

    if (!last_frame_timestamps.empty() && sculpt_manager->has_performed_evaluation()) {
        last_evaluation_time = last_frame_timestamps[0];
    }
}
