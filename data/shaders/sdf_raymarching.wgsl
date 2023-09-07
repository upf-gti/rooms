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

};

struct SdfData {
    data : array<vec4f>
};

@group(0) @binding(0) var left_eye_texture: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(1) var right_eye_texture: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(2) var<storage, read_write> sdf_data : SdfData;

@group(1) @binding(0) var<uniform> compute_data : ComputeData;
@group(1) @binding(1) var<uniform> preview_edit : Edit;

const MAX_DIST = 100.0;
const MIN_HIT_DIST = 0.002;
const DERIVATIVE_STEP = 1.0 / 512.0;

const ambientCoeff = 0.2;
const diffuseCoeff = 0.9;
const specularCoeff = 1.0;
const specularExponent = 64.0;
const lightPos = vec3f(0.0, 2.0, 1.0);

const fov = 45.0;
const up = vec3f(0.0, 1.0, 0.0);

fn sample_sdf(position : vec3f, trilinear : bool) -> Surface
{
    let p = (position - compute_data.sculpt_start_position) * 512.0 + vec3f(256.0) - vec3f(0.0, 512.0, 0.0);

    if (p.x < 0.0 || p.x > 511 ||
        p.y < 0.0 || p.y > 511 ||
        p.z < 0.0 || p.z > 511) {

        var surface : Surface = Surface(vec3(0.0, 0.0, 0.0), 0.01);
        // surface = add_preview_edit(p + compute_data.sculpt_start_position * 512.0, surface);
        return surface;
    }

    var data : vec4f;

    // From: http://paulbourke.net/miscellaneous/interpolation/
    if (trilinear) {
        let x_f : f32 = abs(fract(p.x));
        let y_f : f32 = abs(fract(p.y));
        let z_f : f32 = abs(fract(p.z));

        let x : u32 = u32(floor(p.x));
        let y : u32 = u32(floor(p.y));
        let z : u32 = u32(floor(p.z));

        let index000 : u32 = x + y * 512u + z * 512u * 512u;
        let index100 : u32 = (x + 1) + (y + 0) * 512u + (z + 0) * 512u * 512u;
        let index010 : u32 = (x + 0) + (y + 1) * 512u + (z + 0) * 512u * 512u;
        let index001 : u32 = (x + 0) + (y + 0) * 512u + (z + 1) * 512u * 512u;
        let index101 : u32 = (x + 1) + (y + 0) * 512u + (z + 1) * 512u * 512u;
        let index011 : u32 = (x + 0) + (y + 1) * 512u + (z + 1) * 512u * 512u;
        let index110 : u32 = (x + 1) + (y + 1) * 512u + (z + 0) * 512u * 512u;
        let index111 : u32 = (x + 1) + (y + 1) * 512u + (z + 1) * 512u * 512u;

        data = sdf_data.data[index000] * (1.0 - x_f) * (1.0 - y_f) * (1.0 - z_f) +
                        sdf_data.data[index100] * x_f * (1.0 - y_f) * (1.0 - z_f) +
                        sdf_data.data[index010] * (1.0 - x_f) * y_f * (1.0 - z_f) +
                        sdf_data.data[index001] * (1.0 - x_f) * (1.0 - y_f) * z_f +
                        sdf_data.data[index101] * x_f * (1.0 - y_f) * z_f +
                        sdf_data.data[index011] * (1.0 - x_f) * y_f * z_f +
                        sdf_data.data[index110] * x_f * y_f * (1.0 - z_f) +
                        sdf_data.data[index111] * x_f * y_f * z_f;
    } else {
        let x : u32 = u32(round(p.x));
        let y : u32 = u32(round(p.y));
        let z : u32 = u32(round(p.z));

        let index : u32 = x + y * 512u + z * 512u * 512u;
        data = sdf_data.data[index];
    }

    var surface : Surface = Surface(data.xyz, data.w);

    surface = add_preview_edit(p + compute_data.sculpt_start_position * 512.0, surface);

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

fn blinn_phong(rayOrigin : vec3f, position : vec3f, lightPosition : vec3f, ambient : vec3f, diffuse : vec3f) -> vec3f
{
    let normal : vec3f = estimate_normal(position);
    let toEye : vec3f = normalize(rayOrigin - position);
    let toLight : vec3f = normalize(lightPosition - position);
    // let reflection : vec3f = reflect(-toLight, normal); // uncomment for Phong model
    let halfwayDir : vec3f = normalize(toLight + toEye);

    let ambientFactor : vec3f = ambient * ambientCoeff * diffuse;
    let diffuseFactor : vec3f = diffuse * max(0.0, dot(normal, toLight));
    // let specularFactor : vec3f = diffuse * pow(max(0.0, dot(toEye, reflection)), specularExponent)
    //                     * specularCoeff; // uncomment for Phong model
    let specularFactor : vec3f = diffuse * pow(max(0.0, dot(normal, halfwayDir)), specularExponent)
                        * specularCoeff;

    return ambientFactor + diffuseFactor + specularFactor;
}

fn raymarch(rayOrigin : vec3f, rayDir : vec3f, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4, 0.4, 0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var surface_min_dist : f32 = 100.0;
    var surface : Surface;
	for (var i : i32 = 0; depth < MAX_DIST && i < 250; i++)
	{
		let pos = rayOrigin + rayDir * depth;

        surface = sample_sdf(pos, surface_min_dist < 0.01);

		if (surface.distance < MIN_HIT_DIST && surface.distance > -MIN_HIT_DIST) {
            let proj_pos : vec4f = view_proj * vec4f(pos, 1.0);
            depth = proj_pos.z / proj_pos.w;
			return vec4f(blinn_phong(rayOrigin, pos, lightPos + lightOffset, ambientColor, surface.color), depth);
		}
		depth += surface.distance;
        surface_min_dist = surface.distance;
	}
    return vec4f(1.0, 1.0, 1.0, 1.0);
}

fn get_ray_direction(inv_view_projection : mat4x4f, uv : vec2f) -> vec3f
{
	// convert coordinates from [0, 1] to [-1, 1]
	var screenCoord : vec4f = vec4f((uv - 0.5) * 2.0, 1.0, 1.0);

	var rayDir : vec4f = inv_view_projection * screenCoord;
    rayDir = rayDir / rayDir.w;

	return normalize(rayDir.xyz);
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
