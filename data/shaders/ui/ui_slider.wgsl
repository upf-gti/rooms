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
    color  : vec4f,
};

struct InstanceData {
    data : array<RenderMeshData>
}

struct CameraData {
    view_projection : mat4x4f,
};

struct UIData {
    is_hovered : f32,
    num_group_items : f32,
    is_selected : f32,
    is_color_button : f32,
    picker_color: vec4f,
    slider_value : f32,
    dummy0 : f32,
    dummy1 : f32,
    dummy2 : f32,
};

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
@group(2) @binding(1) var texture_sampler : sampler;

@group(3) @binding(0) var<uniform> ui_data : UIData;

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

fn remap_range(oldValue : f32, oldMin: f32, oldMax : f32, newMin : f32, newMax : f32) -> f32 {
    return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv);

    if (color.a < 0.01) {
        discard;
    }

    var value = ui_data.slider_value;

    // Mask
    var uvs = in.uv;
    var button_size = 32.0;
    var tx = max(button_size, 32.0 * ui_data.num_group_items);
    var divisions = tx / button_size;
    uvs.x *= divisions;
    uvs.y = 1.0 - uvs.y;
    var p = vec2f(clamp(uvs.x, 0.5, divisions - 0.5), 0.5);
    var d = 1.0 - step(0.45, distance(uvs, p));

    // add gradient at the end to simulate the slider thumb
    var mesh_color = in.color;

    var axis = select( in.uv.x, uvs.y, ui_data.num_group_items == 1.0 );
    var grad = smoothstep(value, 1.0, axis / value);
    grad = pow(grad, 12.0);
    mesh_color += grad * 0.4;

    let back_color = vec3f(0.3);
    var final_color = select( mesh_color, back_color, axis > value || d < 1.0 );

    let hover_color = vec3f(0.95, 0.76, 0.17);
    final_color = select( final_color, hover_color, d < 1.0 && ui_data.is_hovered > 0.0 );

    out.color = vec4f(pow(final_color, vec3f(2.2)), color.a);
    return out;
}