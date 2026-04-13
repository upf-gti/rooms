#include "engine/engine.h"
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

void get_engine_config(sEngineConfiguration& out_config)
{
    out_config.window_width = 1280;
    out_config.window_height = 720;

    out_config.window_title = "ROOMS";
    //out_config.fullscreen = true;

    sRendererConfiguration rooms_render_config = {};

#ifndef DISABLE_RAYMARCHER
    rooms_render_config.required_limits.maxBufferSize = 536870912;
    rooms_render_config.required_limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float);
    rooms_render_config.required_limits.maxStorageBuffersPerShaderStage = 8; // GTX 1080 friendly :(
    rooms_render_config.required_limits.maxComputeInvocationsPerWorkgroup = 512;
#endif

    out_config.custom_engine_instance = new RoomsEngine();
    out_config.custom_renderer_instance = new RoomsRenderer(rooms_render_config);

    // Optional callbacks
    out_config.engine_post_initialize = nullptr;
    out_config.engine_pre_update = nullptr;
    out_config.engine_post_update = nullptr;
    out_config.engine_render = nullptr;
}
