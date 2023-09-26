#include "engine/rooms_engine.h"
#include "graphics/raymarching_renderer.h"

#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
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

#endif

void closeWindow(GLFWwindow* window) {
#if !defined(XR_SUPPORT) || (defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW))
    glfwDestroyWindow(window);
    glfwTerminate();
#endif
}

bool shouldClose(bool use_glfw, GLFWwindow* window) {
    return glfwWindowShouldClose(window);
}

int main() {

    RoomsEngine* engine = new RoomsEngine();
    RaymarchingRenderer* raymarching_renderer = new RaymarchingRenderer();
    GLFWwindow* window = nullptr;

    WGPURequiredLimits required_limits = {};
    required_limits.limits.maxVertexAttributes = 4;
    required_limits.limits.maxVertexBuffers = 1;
    required_limits.limits.maxBindGroups = 2;
    required_limits.limits.maxUniformBuffersPerShaderStage = 1;
    required_limits.limits.maxUniformBufferBindingSize = 32 * 4 * sizeof(float);
    required_limits.limits.minUniformBufferOffsetAlignment = 256;
    required_limits.limits.minStorageBufferOffsetAlignment = 256;
#ifndef DISABLE_RAYMARCHER
    required_limits.limits.maxBufferSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4;
    required_limits.limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float) * 4;
    required_limits.limits.maxComputeInvocationsPerWorkgroup = 512;
#endif
    required_limits.limits.maxSamplersPerShaderStage = 1;

    raymarching_renderer->set_required_limits(required_limits);

#ifdef __EMSCRIPTEN__
    int screen_width = canvas_get_width();
    int screen_height = canvas_get_height();

    emscripten_set_resize_callback(
        EMSCRIPTEN_EVENT_TARGET_WINDOW,
        (void*)engine, 0, on_web_display_size_changed
    );

#elif defined(XR_SUPPORT)
    // Keep XR aspect ratio
    int screen_width = 992;
    int screen_height = 1000;
#else
    int screen_width = 1280;
    int screen_height = 720;
#endif

    const bool use_xr = raymarching_renderer->get_openxr_available();
    const bool use_mirror_screen = engine->get_use_mirror_window();

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

    if (engine->initialize(raymarching_renderer, window, use_glfw, use_mirror_screen)) {
        std::cout << "Could not initialize engine" << std::endl;
        closeWindow(window);
        return 1;
    }

    std::cout << "Engine initialized" << std::endl;

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
    }
#endif

    engine->clean();

    closeWindow(window);

    delete engine;
    delete raymarching_renderer;

    return 0;
}
