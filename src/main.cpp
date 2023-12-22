#include "engine/rooms_engine.h"
#include "graphics/renderers/rooms_renderer.h"

#include <GLFW/glfw3.h>

#include "spdlog/spdlog.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/bind.h>
EM_JS(int, canvas_get_width, (), {
  return canvas.clientWidth;
});
EM_JS(int, canvas_get_height, (), {
  return canvas.clientHeight;
});
static EM_BOOL on_web_display_size_changed(int event_type,
    const EmscriptenUiEvent* ui_event, void* user_data)
{
    RoomsEngine* engine = reinterpret_cast<RoomsEngine*>(user_data);
    engine->resize_window(ui_event->windowInnerWidth, ui_event->windowInnerHeight);
    return true;
}
// Binding code
EMSCRIPTEN_BINDINGS(_Class_) {
    emscripten::class_<RoomsEngine>("Engine")
        .constructor<>()
        .class_function("setEnvironment", &RoomsEngine::set_skybox_texture)
        .class_function("loadGLB", &RoomsEngine::load_glb)
        .class_function("toggleSceneRotation", &RoomsEngine::toggle_rotation);
}
#endif

void closeWindow(GLFWwindow* window)
{
#if !defined(XR_SUPPORT) || (defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW))
    glfwDestroyWindow(window);
    glfwTerminate();
#endif
}

bool shouldClose(bool use_glfw, GLFWwindow* window)
{
    return glfwWindowShouldClose(window);
}

int main()
{
    spdlog::set_pattern("[%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);

    RoomsEngine* engine = new RoomsEngine();
    RoomsRenderer* renderer = new RoomsRenderer();
    GLFWwindow* window = nullptr;

    WGPURequiredLimits required_limits = {};
    required_limits.limits.maxVertexAttributes = 4;
    required_limits.limits.maxVertexBuffers = 1;
    required_limits.limits.maxBindGroups = 2;
    required_limits.limits.maxUniformBuffersPerShaderStage = 1;
    required_limits.limits.maxUniformBufferBindingSize = 65536;
    required_limits.limits.minUniformBufferOffsetAlignment = 256;
    required_limits.limits.minStorageBufferOffsetAlignment = 256;
#ifndef DISABLE_RAYMARCHER
    required_limits.limits.maxBufferSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4;
    required_limits.limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4;
    required_limits.limits.maxStorageBuffersPerShaderStage = 8; // GTX 1080 friendly :(
    required_limits.limits.maxComputeInvocationsPerWorkgroup = 1024;
#else
    required_limits.limits.maxComputeInvocationsPerWorkgroup = 256;
#endif
    required_limits.limits.maxSamplersPerShaderStage = 1;
    required_limits.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;

    renderer->set_required_limits(required_limits);

#ifdef __EMSCRIPTEN__
    int screen_width = canvas_get_width();
    int screen_height = canvas_get_height();

    emscripten_set_resize_callback(
        EMSCRIPTEN_EVENT_TARGET_WINDOW,
        (void*)engine, 0, on_web_display_size_changed
    );
#else
    int screen_width = 1600;
    int screen_height = 900;
#endif

    const bool use_xr = renderer->get_openxr_available();
    const bool use_mirror_screen = engine->get_use_mirror_window();

    if (use_xr) {
        // Keep XR aspect ratio
        screen_width = 992;
        screen_height = 1000;
    }

    // Only init glfw if no xr or using mirror
    const bool use_glfw = !use_xr || (use_xr && use_mirror_screen);

    if (use_glfw) {
        if (!glfwInit()) {
            return 1;
        }

        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(screen_width, screen_height, "WebGPU Engine", NULL, NULL);
    }

    if (engine->initialize(renderer, window, use_glfw, use_mirror_screen)) {
        spdlog::error("Could not initialize engine");
        closeWindow(window);
        return 1;
    }

    spdlog::info("Engine initialized");

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(
        [](void* userData) {
            RoomsEngine* engine = reinterpret_cast<RoomsEngine*>(userData);
            engine->on_frame();
        },
        (void*)engine,
        0, true
    );

#else
    while (!shouldClose(use_glfw, window)) {
        engine->on_frame();

        std::string fps = "FPS: " + std::to_string(1.0 / (std::max(engine->get_delta_time(), 0.0001f)));
        glfwSetWindowTitle(renderer->get_glfw_window(), fps.c_str());
    }
#endif

    engine->clean();

    closeWindow(window);

    delete engine;
    delete renderer;

    return 0;
}
