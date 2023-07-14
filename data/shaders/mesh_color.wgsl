struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec3f,

};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
};

struct RenderMeshData {
    model  : mat4x4f,
    color  : vec3f,
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(0) var<uniform> mesh_data : RenderMeshData;
@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera_data.view_projection * mesh_data.model * vec4f(in.position, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * mesh_data.color;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    out.color = vec4f(in.color, 1.0); // Color

    return out;
}