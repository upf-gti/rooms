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

fn modulo_euclidean(a: f32, b: f32) -> f32
{
	var m = a % b;
	if (m < 0.0) {
		if (b < 0.0) {
			m -= b;
		} else {
			m += b;
		}
	}
	return m;
}

fn modulo_euclidean_vec3( v : vec3f, m: f32) -> vec3f
{
	return vec3f(
        modulo_euclidean(v.x, m),
        modulo_euclidean(v.y, m),
        modulo_euclidean(v.z, m)
    );
}

fn hsv2rgb_smooth( c : vec3f ) -> vec3f
{
    var m = modulo_euclidean_vec3(c.x * 6.0 + vec3f(0.0, 4.0, 2.0), 6.0);
    var rgb = clamp( abs(m - 3.0) - 1.0, vec3f(0.0), vec3f(1.0) );

	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing	

	return mix(vec3(1.0),mix( vec3(1.0), rgb, c.y), c.z);
}

fn getColor( uvs : vec2f ) -> vec3f
{
    let pi = 3.1415927;

    var p = uvs;
    
    var r = pi / 2.0;

    p *= mat2x2f(cos(r),sin(r),-sin(r), cos(r));

    var polar = vec2f(atan2(p.y, p.x), length(p));
    
    var percent = (polar.x + pi) / (2.0 * pi);
    
    var hsv = vec3f(percent, 1., polar.y);
    
    return hsv2rgb_smooth(hsv);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv);

    if (color.a < 0.01) {
        discard;
    }

    // Mask
    var uvs = in.uv;
    var divisions = 1.0;
    uvs.x *= divisions;
    var p = vec2f(clamp(uvs.x, 0.5, 0.5), 0.5);
    var d = 1.0 - step(0.435, distance(uvs, p));

    uvs = in.uv;
    var current_color = ui_data.picker_color.rgb * ui_data.picker_color.a;
    var final_color = getColor(uvs * 2 - 1) * d + current_color * (1 - d);

    out.color = vec4f(pow(final_color, vec3f(2.2)), color.a);
    return out;
}