struct VertexInput {
    @builtin(instance_index) instance_id : u32,
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

struct ComputeData {
    view_projection_left_eye  : mat4x4f,
    view_projection_right_eye : mat4x4f,

    inv_view_projection_left_eye  : mat4x4f,
    inv_view_projection_right_eye : mat4x4f,

    left_eye_pos    : vec3f,
    render_height   : f32,
    right_eye_pos   : vec3f,
    render_width    : f32,

    time            : f32,
    camera_near     : f32,
    camera_far      : f32,
    dummy0          : f32,

    sculpt_start_position   : vec3f,
    dummy1                  : f32,

    sculpt_rotation : vec4f

};

@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var<storage, read_write> proxy_box_position_buffer: array<vec3f>;

@group(1) @binding(0) var<uniform> compute_data : ComputeData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    out.position = camera_data.view_projection * instance_data.model * vec4f(in.position, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * instance_data.color.rgb;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    out.color = vec4f(pow(in.color, vec3f(2.2)), 1.0); // Color

    return out;
}