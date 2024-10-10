#include "rooms_renderer.h"

#include "graphics/shader.h"

RoomsRenderer::RoomsRenderer() : Renderer()
{

}

RoomsRenderer::~RoomsRenderer()
{

}

int RoomsRenderer::initialize(GLFWwindow* window, bool use_mirror_screen)
{
    Renderer::initialize(window, use_mirror_screen);

    Shader::set_custom_define("SDF_RESOLUTION", SDF_RESOLUTION);
    Shader::set_custom_define("SCULPT_MAX_SIZE", SCULPT_MAX_SIZE);

    clear_color = glm::vec4(0.22f, 0.22f, 0.22f, 1.0);

    raymarching_renderer.initialize(use_mirror_screen);

    set_custom_pass_user_data(&raymarching_renderer);

#ifndef DISABLE_RAYMARCHER
    custom_post_opaque_pass = [](void* user_data, WGPURenderPassEncoder render_pass, uint32_t camera_stride_offset = 0) {
        RaymarchingRenderer* raymarching_renderer = reinterpret_cast<RaymarchingRenderer*>(user_data);
        raymarching_renderer->render_raymarching_proxy(render_pass, camera_stride_offset);
    };
#endif

    return 0;
}

void RoomsRenderer::clean()
{
    Renderer::clean();

    raymarching_renderer.clean();
}

void RoomsRenderer::update(float delta_time)
{
    Renderer::update(delta_time);

    raymarching_renderer.update_sculpt(global_command_encoder);
}

void RoomsRenderer::render()
{
    Renderer::render();

#ifndef __EMSCRIPTEN__
    last_frame_timestamps = get_timestamps();

    if (!last_frame_timestamps.empty() && raymarching_renderer.has_performed_evaluation()) {
        last_evaluation_time = last_frame_timestamps[0];
    }
#endif
}
