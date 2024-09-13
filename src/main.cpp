#include "engine/rooms_engine.h"
#include "graphics/renderers/rooms_renderer.h"

#include "spdlog/spdlog.h"

int main()
{
    spdlog::set_pattern("[%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);

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

    if (engine->initialize(renderer)) {
        return 1;
    }

    engine->start_loop();

    engine->clean();

    delete engine;
    delete renderer;

    return 0;
}
