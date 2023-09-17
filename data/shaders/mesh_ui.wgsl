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

struct RenderMeshData {
    model  : mat4x4f,
    color  : vec4f
};

struct InstanceData {
    data : array<RenderMeshData>
}

struct CameraData {
    view_projection : mat4x4f
};

struct UIData {
    dummy : vec3f,
    num_group_items : f32
};

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;
@group(1) @binding(0) var<uniform> camera_data : CameraData;
@group(2) @binding(0) var<uniform> ui_data : UIData;

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

    var uvs = in.uv;
    var button_size = 32.0;
    // var items = 2.0;
    var tx = max(button_size, 32.0 * ui_data.num_group_items);
    var divisions = tx / button_size;
    uvs.x *= divisions;
    var p = vec2f(clamp(uvs.x, 0.5, divisions - 0.5), 0.5);
    var d = 1.0 - step(0.5, distance(uvs, p));

    out.color = vec4f(pow(in.color * d, vec3f(2.2)), d);
    return out;
}