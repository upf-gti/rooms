#include "engine/rooms_engine.h"
#include "graphics/renderers/rooms_renderer.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/bind.h>

// Binding code
EMSCRIPTEN_BINDINGS(_Class_) {

    emscripten::class_<RoomsEngine>("Engine")
        .constructor<>()
        .class_function("getInstance", &RoomsEngine::get_instance, emscripten::return_value_policy::reference())
        .function("setWasmModuleInitialized", &RoomsEngine::set_wasm_module_initialized);

}
#endif

int main()
{
    RoomsEngine* engine = new RoomsEngine();

    sRendererConfiguration rooms_render_config;

#ifndef DISABLE_RAYMARCHER
    rooms_render_config.required_limits.maxBufferSize = 536870912;
    rooms_render_config.required_limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float);
    rooms_render_config.required_limits.maxStorageBuffersPerShaderStage = 8; // GTX 1080 friendly :(
    rooms_render_config.required_limits.maxComputeInvocationsPerWorkgroup = 512;
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
