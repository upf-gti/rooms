#include sdf_functions.wgsl

struct ComputeData {
    inv_view_projection_left_eye  : mat4x4f,
    inv_view_projection_right_eye : mat4x4f,

    left_eye_pos    : vec3f,
    render_height   : f32,
    right_eye_pos   : vec3f,
    render_width    : f32,

    time            : f32,
    dummy0          : f32,
    dummy1          : f32,
    dummy2          : f32,
};

@group(0) @binding(0) var left_eye_texture: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(1) var right_eye_texture: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(2) var<storage, read_write> sdf_data : SdfData;

@group(1) @binding(0) var<uniform> compute_data : ComputeData;

const MAX_DIST = 400.0;
const MIN_HIT_DIST = 0.0001;
const DERIVATIVE_STEP = 1.0 / 512.0;

const ambientCoeff = 0.2;
const diffuseCoeff = 0.9;
const specularCoeff = 1.0;
const specularExponent = 64.0;
const lightPos = vec3f(0.0, 2.0, 1.0);

const fov = 45.0;
const up = vec3f(0.0, 1.0, 0.0);

fn sampleSdf(position : vec3f) -> Surface
{
    let p = position * 512.0 + vec3f(256.0);// - vec3f(0.0, 500.0, 0.0);

    if (p.x < 0.0 || p.x > 511 ||
        p.y < 0.0 || p.y > 511 ||
        p.z < 0.0 || p.z > 511) {
        return Surface(vec3(0.0, 0.0, 0.0), 0.01);
    }

    let x : u32 = u32(round(p.x));
    let y : u32 = u32(round(p.y));
    let z : u32 = u32(round(p.z));

    let index : u32 = x + y * 512u + z * 512u * 512u;
    return sdf_data.data[index];
}

fn estimateNormal(p : vec3f) -> vec3f
{
    return normalize(vec3f(
        sampleSdf(vec3f(p.x + DERIVATIVE_STEP, p.y, p.z)).distance - sampleSdf(vec3f(p.x - DERIVATIVE_STEP, p.y, p.z)).distance,
        sampleSdf(vec3f(p.x, p.y + DERIVATIVE_STEP, p.z)).distance - sampleSdf(vec3f(p.x, p.y - DERIVATIVE_STEP, p.z)).distance,
        sampleSdf(vec3f(p.x, p.y, p.z + DERIVATIVE_STEP)).distance - sampleSdf(vec3f(p.x, p.y, p.z - DERIVATIVE_STEP)).distance
    ));
}

fn blinnPhong(rayOrigin : vec3f, position : vec3f, lightPosition : vec3f, ambient : vec3f, diffuse : vec3f) -> vec3f
{
    let normal : vec3f = estimateNormal(position);
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

fn raymarch(rayOrigin : vec3f, rayDir : vec3f) -> vec3f
{
    let ambientColor = vec3f(0.4, 0.4, 0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth = 0.0;
	for (var i : i32 = 0; depth < MAX_DIST && i < 500; i++)
	{
		let pos = rayOrigin + rayDir * depth;
        let surface : Surface = sampleSdf(pos);
		if (surface.distance < MIN_HIT_DIST) {
			return blinnPhong(rayOrigin, pos, lightPos + lightOffset, ambientColor, surface.color);
		}
		depth += surface.distance;
	}
    return missColor;
}

fn getRayDirection(inv_view_projection : mat4x4f, uv : vec2f) -> vec3f
{
	// convert coordinates from [0, 1] to [-1, 1]
	var screenCoord : vec4f = vec4f((uv - 0.5) * 2.0, 1.0, 1.0);

	var rayDir : vec4f = inv_view_projection * screenCoord;
    rayDir = rayDir / rayDir.w;

	return normalize(rayDir.xyz);
}

@compute @workgroup_size(16, 16, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {

    let pixel_size = 1.0 / vec2f(compute_data.render_width, compute_data.render_height);
    var uv = vec2f(id.xy) * pixel_size;
    //uv.y = 1.0 - uv.y;
    let ray_dir_left = getRayDirection(compute_data.inv_view_projection_left_eye, uv);
    let ray_dir_right = getRayDirection(compute_data.inv_view_projection_right_eye, uv);

    textureStore(left_eye_texture, id.xy, vec4f(raymarch(compute_data.left_eye_pos, ray_dir_left), 1.0));
    textureStore(right_eye_texture, id.xy, vec4f(raymarch(compute_data.right_eye_pos, ray_dir_right), 1.0));
}