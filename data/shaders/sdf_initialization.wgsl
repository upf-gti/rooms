#include sdf_functions.wgsl

struct SdfData {
    data : array<Surface>
};

@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 8)

fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(write_sdf, id, vec4f(0.0, 0.0, 0.0, 0.025));
}