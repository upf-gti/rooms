#include ../mesh_includes.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
@group(2) @binding(7) var texture_sampler : sampler;

@group(3) @binding(0) var<uniform> ui_data : UIData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    var world_position = instance_data.model * vec4f(in.position, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera_data.view_projection * world_position;
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

	rgb = rgb * rgb * (3.0 - 2.0 * rgb); // cubic smoothing

	return mix(vec3(1.0),mix( vec3(1.0), rgb, c.y), c.z);
}

fn getColor( uvs : vec2f ) -> vec3f
{
    let pi = 3.14159265359;

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

    var dummy = camera_data.eye;

    var out: FragmentOutput;

    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv);
    color = pow(color, vec4f(2.2));

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
    var current_color = pow(ui_data.picker_color.rgb, vec3f(2.2)) * ui_data.picker_color.a;
    var final_color = pow(getColor(uvs * 2 - 1), vec3f(2.2)) * d + current_color * (1 - d);

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3f(1.0 / 2.2));
    }

    out.color = vec4f(final_color, color.a);

    return out;
}