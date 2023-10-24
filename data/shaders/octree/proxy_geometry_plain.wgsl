#include ../sdf_functions.wgsl

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
    @location(3) world_pos : vec3f,
    @location(4) voxel_pos : vec3f,
    @location(5) voxel_center : vec3f
};

struct SculptData {
    sculpt_start_position   : vec3f,
    dummy1                  : f32,
    sculpt_rotation         : vec4f,
    sculpt_inv_rotation     : vec4f
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(2) var texture_sampler : sampler;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(6) var<storage, read> proxy_box_position_buffer: array<vec3f>;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

// 1 / SDF_SIZE * 8 (texels that compose a brick) / 2 (the cube is centered, so its the halfsize) = 0.0078125
const BOX_SIZE : f32 = ((1.0 / SDF_RESOLUTION) * 8.0) / 2.0;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_pos : vec3f = proxy_box_position_buffer[in.instance_id];

    var voxel_pos : vec3f = in.position * BOX_SIZE + instance_pos;
    var world_pos : vec3f = rotate_point_quat(voxel_pos, sculpt_data.sculpt_rotation);
    world_pos += sculpt_data.sculpt_start_position;

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    // world_pos = vec4f(rotate_point_quat(world_pos.xyz, sculpt_data.sculpt_rotation), 1.0);
    out.position = camera_data.view_projection * vec4f(world_pos, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color;
    out.normal = in.normal;
    out.world_pos = world_pos.xyz;
    out.voxel_pos = voxel_pos;
    out.voxel_center = instance_pos;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

@group(0) @binding(1) var<uniform> eye_position : vec3f;
@group(2) @binding(0) var<uniform> sculpt_data : SculptData;

const VOXEL_SIZE = vec3f(8.0 * (1.0 / SDF_RESOLUTION));
const MAX_DIST = sqrt(3.0) * 8.0 * (1.0 / SDF_RESOLUTION);
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

fn ray_AABB_intersection_distance(ray_origin : vec3f,
                                  ray_dir : vec3f,
                                  box_origin : vec3f,
                                  box_size : vec3f) -> f32 {
    let box_min : vec3f = box_origin - (box_size / 2.0);
    let box_max : vec3f = box_min + box_size;

    // Testing X axis slab
    let tx1 : f32 = (box_min.x - ray_origin.x) / ray_dir.x;
    let tx2 : f32 = (box_max.x - ray_origin.x) / ray_dir.x;
    var tmin : f32 = min(tx1, tx2);
    var tmax :f32 = max(tx1, tx2);

    // Testing Y axis slab
    let ty1 : f32 = (box_min.y - ray_origin.y) / ray_dir.y;
    let ty2 : f32= (box_max.y - ray_origin.y) / ray_dir.y;
    tmin = max(min(ty1, ty2), tmin);
    tmax = min(max(ty1, ty2), tmax);

    // Testing Z axis slab
    let tz1 : f32 = (box_min.z - ray_origin.z) / ray_dir.z;
    let tz2 : f32 = (box_max.z - ray_origin.z) / ray_dir.z;
    tmin = max(min(tz1, tz2), tmin);
    tmax = min(max(tz1, tz2), tmax);

    return tmax - tmin;
}

fn sample_sdf(position : vec3f) -> Surface
{
    let rot_p = rotate_point_quat(position - sculpt_data.sculpt_start_position, sculpt_data.sculpt_inv_rotation) + vec3f(0.5);

    if (rot_p.x < 0.0 || rot_p.x > 1.0 ||
        rot_p.y < 0.0 || rot_p.y > 1.0 ||
        rot_p.z < 0.0 || rot_p.z > 1.0) {
        return Surface(vec3(0.0, 0.0, 0.0), 0.01);
    }

    var data : vec4f;

    data = textureSampleLevel(read_sdf, texture_sampler, rot_p, 0.0);

    var surface : Surface = Surface(data.xyz, data.w);

    //surface = add_preview_edits((p  + compute_data.sculpt_start_position) * SDF_RESOLUTION, surface);

    return surface;
}

fn estimate_normal(p : vec3f) -> vec3f
{
    return normalize(vec3f(
        sample_sdf(vec3f(p.x + DERIVATIVE_STEP, p.y, p.z)).distance - sample_sdf(vec3f(p.x - DERIVATIVE_STEP, p.y, p.z)).distance,
        sample_sdf(vec3f(p.x, p.y + DERIVATIVE_STEP, p.z)).distance - sample_sdf(vec3f(p.x, p.y - DERIVATIVE_STEP, p.z)).distance,
        sample_sdf(vec3f(p.x, p.y, p.z + DERIVATIVE_STEP)).distance - sample_sdf(vec3f(p.x, p.y, p.z - DERIVATIVE_STEP)).distance
    ));
}

fn blinn_phong(toEye : vec3f, position : vec3f, lightPosition : vec3f, ambient : vec3f, diffuse : vec3f) -> vec3f
{
    let normal : vec3f = estimate_normal(position);
    let toLight : vec3f = normalize(lightPosition - position);
    let reflection : vec3f = normalize(reflect(-toLight, normal)); // uncomment for Phong model
    let halfwayDir : vec3f = normalize(toLight + toEye);

    let ambientFactor : vec3f = ambient * diffuse;
    let diffuseFactor : vec3f = 0.4 * diffuse * max(0.0, dot(normal, toLight));
    let specularFactor : vec3f = vec3f(0.3) * pow(max(0.0, dot(toEye, reflection)), specularExponent); // uncomment for Phong model
    //let specularFactor : vec3f = diffuse * pow(max(0.0, dot(normal, halfwayDir)), specularExponent) * specularCoeff;

    return ambientFactor + diffuseFactor + specularFactor;
}

fn raymarch(ray_origin : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var surface_min_dist : f32 = 50.0;
    var surface : Surface;

    var edge_threshold = 0.003;
    var edge : f32 = 0.0;

	for (var i : i32 = 0; depth < max_distance && i < 80; i++)
	{
		let pos = ray_origin + ray_dir * depth;

        surface = sample_sdf(pos);

        // Edge detection
        // if((surface_min_dist < edge_threshold) && (surface.distance > surface_min_dist))
        // {
        //     edge = 1.0;
        // }

		if ((surface.distance) < MIN_HIT_DIST) {
            let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
            let proj_pos : vec4f = view_proj * vec4f(pos + ray_dir * epsilon, 1.0);
            depth = proj_pos.z / proj_pos.w;
			return vec4f(blinn_phong(-ray_dir, pos, lightPos + lightOffset, ambientColor, surface.color * (1.0 - edge)), depth);
		}

        surface_min_dist = (surface.distance);
        depth += (surface.distance);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(irradiance_spherical_harmonics(ray_dir.xzy)* (1.0 - edge), 0.999);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    let ray_dir : vec3f = normalize(in.world_pos.xyz - eye_position);
    let ray_dir_voxel_space : vec3f = normalize(in.voxel_pos - rotate_point_quat(eye_position - sculpt_data.sculpt_start_position, sculpt_data.sculpt_inv_rotation));

    let raymarch_distance : f32 = ray_AABB_intersection_distance(in.voxel_pos, ray_dir_voxel_space, in.voxel_center, VOXEL_SIZE);

    let ray_result = raymarch(in.world_pos.xyz, ray_dir, raymarch_distance, camera_data.view_projection);

    out.color = vec4f(pow(ray_result.rgb, vec3f(2.2, 2.2, 2.2)), 1.0); // Color
    out.depth = ray_result.a;

    // out.color = vec4f(1.0, 0.0, 0.0, 1.0); // Color
    // out.depth = 0.0;

    if (out.depth == 0.999) {
        discard;
    }

    return out;
}