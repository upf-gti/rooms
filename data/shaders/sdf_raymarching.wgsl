#include sdf_functions.wgsl

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

struct SdfData {
    data : array<vec4f>
};

@group(0) @binding(0) var left_eye_texture: texture_storage_2d<rgba32float,write>;
@group(0) @binding(1) var right_eye_texture: texture_storage_2d<rgba32float,write>;
@group(0) @binding(2) var read_sdf: texture_3d<f32>;
@group(0) @binding(3) var texture_sampler : sampler;

@group(1) @binding(0) var<uniform> compute_data : ComputeData;
@group(1) @binding(1) var<uniform> preview_edit : Edit;

const MAX_DIST = 1.5;
const MIN_HIT_DIST = 0.00005;
const DERIVATIVE_STEP = 1.0 / SDF_RESOLUTION;

const specularCoeff = 1.0;
const specularExponent = 4.0;
const lightPos = vec3f(0.0, 2.0, 1.0);

const fov = 45.0;
const up = vec3f(0.0, 1.0, 0.0);

// SKYMAP FUNCTION
fn irradiance_spherical_harmonics(n : vec3f) -> vec3f {
    return vec3f(0.366, 0.363, 0.371)
        + vec3f(0.257, 0.252, 0.263) * (n.y)
        + vec3f(0.108, 0.109, 0.113) * (n.z)
        + vec3f(0.028, 0.044, 0.05) * (n.x)
        + vec3f(0.027, 0.038, 0.036) * (n.y * n.x)
        + vec3f(0.11, 0.11, 0.118) * (n.y * n.z)
        + vec3f(-0.11, -0.113, -0.13) * (3.0 * n.z * n.z - 1.0)
        + vec3f(0.016, 0.018, 0.016) * (n.z * n.x)
        + vec3f(-0.033, -0.033, -0.037) * (n.x * n.x - n.y * n.y);
}

fn sample_sdf(position : vec3f, trilinear : bool) -> Surface
{
    let p = (position - compute_data.sculpt_start_position + vec3(0.5, -0.5, 0.5));

    let rot_p = rotate_point_quat(p - vec3f(0.5), compute_data.sculpt_rotation) + vec3f(0.5);

    if (rot_p.x < 0.0 || rot_p.x > 1.0 ||
        rot_p.y < 0.0 || rot_p.y > 1.0 ||
        rot_p.z < 0.0 || rot_p.z > 1.0) {
        return Surface(vec3(0.0, 0.0, 0.0), 0.01);
    }

    var data : vec4f;

    data = textureSampleLevel(read_sdf, texture_sampler, rot_p, 0.0);

    var surface : Surface = Surface(data.xyz, data.w);

    surface = add_preview_edit(p + compute_data.sculpt_start_position, surface);

    return surface;
}

fn add_preview_edit(position : vec3f, surface : Surface) -> Surface
{
    return evalEdit(position, surface, preview_edit);
}

fn estimate_normal(p : vec3f) -> vec3f
{
    return normalize(vec3f(
        sample_sdf(vec3f(p.x + DERIVATIVE_STEP, p.y, p.z), true).distance - sample_sdf(vec3f(p.x - DERIVATIVE_STEP, p.y, p.z), true).distance,
        sample_sdf(vec3f(p.x, p.y + DERIVATIVE_STEP, p.z), true).distance - sample_sdf(vec3f(p.x, p.y - DERIVATIVE_STEP, p.z), true).distance,
        sample_sdf(vec3f(p.x, p.y, p.z + DERIVATIVE_STEP), true).distance - sample_sdf(vec3f(p.x, p.y, p.z - DERIVATIVE_STEP), true).distance
    ));
}

fn blinn_phong(ray_origin : vec3f, position : vec3f, lightPosition : vec3f, ambient : vec3f, diffuse : vec3f) -> vec3f
{
    let normal : vec3f = estimate_normal(position);
    let toEye : vec3f = normalize(ray_origin - position);
    let toLight : vec3f = normalize(lightPosition - position);
    let reflection : vec3f = normalize(reflect(-toLight, normal)); // uncomment for Phong model
    let halfwayDir : vec3f = normalize(toLight + toEye);

    let ambientFactor : vec3f = ambient * diffuse;
    let diffuseFactor : vec3f = 0.4 * diffuse * max(0.0, dot(normal, toLight));
    let specularFactor : vec3f = vec3f(0.3) * pow(max(0.0, dot(toEye, reflection)), specularExponent); // uncomment for Phong model
    //let specularFactor : vec3f = diffuse * pow(max(0.0, dot(normal, halfwayDir)), specularExponent) * specularCoeff;

    return ambientFactor + diffuseFactor + specularFactor;
}

fn raymarch(ray_origin : vec3f, ray_dir : vec3f, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = clamp(length(ray_origin - compute_data.sculpt_start_position) - 1.412, 0.0, MAX_DIST);
    var surface_min_dist : f32 = 100.0;
    var surface : Surface;
   
	for (var i : i32 = 0; depth < MAX_DIST && i < 200; i++)
	{
		let pos = ray_origin + ray_dir * depth;

        surface = sample_sdf(pos, surface_min_dist < 0.02);

		if (surface.distance < MIN_HIT_DIST) {
            let proj_pos : vec4f = view_proj * vec4f(pos, 1.0);
            depth = proj_pos.z / proj_pos.w;
			return vec4f(blinn_phong(ray_origin, pos, lightPos + lightOffset, ambientColor, surface.color), depth);
		}

        surface_min_dist = surface.distance;
        depth += surface.distance;
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(irradiance_spherical_harmonics(ray_dir), 0.999);
}

fn get_ray_direction(inv_view_projection : mat4x4f, uv : vec2f) -> vec3f
{
	// convert coordinates from [0, 1] to [-1, 1]
	var screenCoord : vec4f = vec4f((uv - 0.5) * 2.0, 1.0, 1.0);

	var ray_dir : vec4f = inv_view_projection * screenCoord;
    ray_dir = ray_dir / ray_dir.w;

	return normalize(ray_dir.xyz);
}

fn map_depths_to_log(depth: f32) -> f32 {
    return log(depth + 1.0) / log(compute_data.camera_far + 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {

    let pixel_size = 1.0 / vec2f(compute_data.render_width, compute_data.render_height);
    var uv = vec2f(id.xy) * pixel_size;
    uv.y = 1.0 - uv.y;
    let ray_dir_left = get_ray_direction(compute_data.inv_view_projection_left_eye, uv);
    let ray_dir_right = get_ray_direction(compute_data.inv_view_projection_right_eye, uv);

    let left_eye_raymarch_result = raymarch(compute_data.left_eye_pos, ray_dir_left, compute_data.view_projection_left_eye);
    let right_eye_raymarch_result = raymarch(compute_data.right_eye_pos, ray_dir_right, compute_data.view_projection_right_eye);

    textureStore(left_eye_texture, id.xy, left_eye_raymarch_result);

    textureStore(right_eye_texture, id.xy, right_eye_raymarch_result);
}
