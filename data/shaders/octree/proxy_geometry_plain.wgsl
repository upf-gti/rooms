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


struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(1) var<uniform> eye_position : vec3f;
@group(0) @binding(2) var texture_sampler : sampler;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(6) var<storage, read> proxy_box_position_buffer: array<vec3f>;

@group(1) @binding(0) var<uniform> camera_data : CameraData;


const BOX_SIZE : f32 = 0.015625;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_pos : vec3f = proxy_box_position_buffer[in.instance_id];

    let eye_pos : vec3f = eye_position;

    let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    out.position = camera_data.view_projection * model_mat * vec4f(in.position, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    let pos : vec4f = textureSampleLevel(read_sdf, texture_sampler, in.position.xyz, 0.0);
    out.color = vec4f(pow(vec3f(1.0, 0.0, 0.0), vec3f(2.2)), 1.0); // Color

    return out;
}