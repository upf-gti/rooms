#include sdf_functions.wgsl

struct SdfData {
    data : array<Surface>
};

@group(0) @binding(2) var<storage, read_write> sdf_data : SdfData;

@compute @workgroup_size(8, 8, 8)

fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
    var initial_value: Surface;
    initial_value.color = vec3f(0.0, 0.0, 0.0);
    initial_value.distance = 0.035;

    sdf_data.data[id.x + id.y * 512 + id.z * 512 * 512] = initial_value;
}