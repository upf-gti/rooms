#include mesh_includes.wgsl

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;
@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * instance_data.color.rgb;
    out.normal = (instance_data.model * vec4f(in.normal, 0.0)).xyz;

    var grow_pos = in.position + normalize(in.normal) * 0.001;
    var world_position = instance_data.model * vec4f(grow_pos, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera_data.view_projection * world_position;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var dummy = camera_data.eye;

    var out: FragmentOutput;

    out.color = vec4f(pow(in.color, vec3f(2.2)), 1.0); // Color

    return out;
}