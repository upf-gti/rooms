#include "engine/rooms_engine.h"
#include "graphics/renderers/rooms_renderer.h"

int main()
{
    RoomsEngine* engine = new RoomsEngine();
    RoomsRenderer* renderer = new RoomsRenderer();

    WGPURequiredLimits required_limits = {};
    required_limits.limits.maxVertexAttributes = 4;
    required_limits.limits.maxVertexBuffers = 1;
    required_limits.limits.maxBindGroups = 2;
    required_limits.limits.maxUniformBuffersPerShaderStage = 1;
    required_limits.limits.maxUniformBufferBindingSize = 65536;
    required_limits.limits.minUniformBufferOffsetAlignment = 256;
    required_limits.limits.minStorageBufferOffsetAlignment = 256;
#ifndef DISABLE_RAYMARCHER
    required_limits.limits.maxBufferSize = 536870912;
    required_limits.limits.maxStorageBufferBindingSize = SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION * sizeof(float);
    required_limits.limits.maxStorageBuffersPerShaderStage = 8; // GTX 1080 friendly :(
    required_limits.limits.maxComputeInvocationsPerWorkgroup = 1024;
#else
    required_limits.limits.maxComputeInvocationsPerWorkgroup = 256;
#endif
    required_limits.limits.maxSamplersPerShaderStage = 1;
    required_limits.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;

    renderer->set_required_limits(required_limits);

    // Set the timestamp in all platforms except the web
    std::vector<WGPUFeatureName> required_features;
#if !defined(__EMSCRIPTEN__)
    required_features.push_back(WGPUFeatureName_TimestampQuery);
#endif
    renderer->set_required_features(required_features);

    if (engine->initialize(renderer, { .window_title = "ROOMS" })) {
        return 1;
    }

    engine->start_loop();

    engine->clean();

    delete engine;
    delete renderer;

    return 0;
}
