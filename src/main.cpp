#include "engine/rooms_engine.h"
#include "graphics/renderers/rooms_renderer.h"

int main()
{
    RoomsEngine* engine = new RoomsEngine();

    sRendererConfiguration rooms_render_config;

#ifndef DISABLE_RAYMARCHER
    rooms_render_config.required_limits.limits.maxBufferSize = 536870912;
    rooms_render_config.required_limits.limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float);
    rooms_render_config.required_limits.limits.maxStorageBuffersPerShaderStage = 8; // GTX 1080 friendly :(
    rooms_render_config.required_limits.limits.maxComputeInvocationsPerWorkgroup = 512;
#endif

    RoomsRenderer* renderer = new RoomsRenderer(rooms_render_config);

    sEngineConfiguration configuration;
    configuration.window_title = "ROOMS";
    //configuration.fullscreen = true;

    if (engine->initialize(renderer, configuration)) {
        return 1;
    }

    engine->start_loop();

    engine->clean();

    delete engine;

    delete renderer;

    return 0;
}
