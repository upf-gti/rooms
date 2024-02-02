#include ../pbr_functions.wgsl
#include ../pbr_light.wgsl

const DERIVATIVE_STEP = 0.5 / SDF_RESOLUTION;
const MAX_ITERATIONS = 60;

const specularCoeff = 1.0;
const specularExponent = 4.0;
const lightPos = vec3f(0.0, 2.0, 1.0);

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

// https://community.khronos.org/t/ray-vs-aabb-exit-point-knowing-entry-point/66307/3
fn ray_AABB_intersection_distance(ray_origin : vec3f,
                                  ray_dir : vec3f,
                                  box_origin : vec3f,
                                  box_size : vec3f) -> f32 {
    let box_min : vec3f = box_origin - (box_size / 2.0);
    let box_max : vec3f = box_min + box_size;

    let min_max : array<vec3f, 2> = array<vec3f, 2>(box_min, box_max);

    var tmax : vec3f;
	let div : vec3f = 1.0 / ray_dir;
	let indexes : vec3i = vec3i(i32(step(0.0, div.x)), i32((step(0.0, div.y))), i32(step(0.0, div.z)));
	tmax.x = (min_max[indexes[0]].x - ray_origin.x) * div.x;
	tmax.y = (min_max[indexes[1]].y - ray_origin.y) * div.y;
	tmax.z = (min_max[indexes[2]].z - ray_origin.z) * div.z;

	return min(min(tmax.x, tmax.y), tmax.z);
}

// TODO: if diffuse variable is not used, performance is increased by 20% (????)
fn apply_light(toEye : vec3f, position : vec3f, position_world : vec3f, normal_i : vec3f, lightPosition : vec3f, material : Material) -> vec3f
{
    //var normal : vec3f = estimate_normal(position, position_world);
    let normal : vec3f = normalize(rotate_point_quat(normal_i, sculpt_data.sculpt_rotation));

    let toLight : vec3f = normalize(lightPosition - position_world);

    var m : LitMaterial;

    m.pos = position_world;
    m.normal = normal;
    m.view_dir = normalize(rotate_point_quat(toEye, sculpt_data.sculpt_rotation));
    m.reflected_dir = reflect( -m.view_dir, m.normal);

    // Material properties

    m.albedo = material.albedo;
    m.metallic = material.metalness;
    m.roughness = max(material.roughness, 0.04);

    m.c_diff = mix(m.albedo, vec3f(0.0), m.metallic);
    m.f0 = mix(vec3f(0.04), m.albedo, m.metallic);
    m.ao = 1.0;

    // var distance : f32 = length(light_position - m.pos);
    // var attenuation : f32 = pow(1.0 - saturate(distance/light_max_radius), 1.5);
    var final_color : vec3f = vec3f(0.0); 
    // final_color += get_direct_light(m, vec3f(1.0), 1.0);

    final_color += tonemap_filmic(get_indirect_light(m), 1.0);

    return final_color;
    //return normal;
    //return diffuse;
    // return vec3f(m.roughness);

}

// https://iquilezles.org/articles/normalsSDF/
fn estimate_normal(sculpt_position : vec3f, atlas_position : vec3f) -> vec3f
{
    let k : vec2f = vec2f(1.0, -1.0);
    return normalize( k.xyy * sample_sdf( sculpt_position + k.xyy * DERIVATIVE_STEP, atlas_position) + 
                      k.yyx * sample_sdf( sculpt_position + k.yyx * DERIVATIVE_STEP, atlas_position ) + 
                      k.yxy * sample_sdf( sculpt_position + k.yxy * DERIVATIVE_STEP, atlas_position) + 
                      k.xxx * sample_sdf( sculpt_position + k.xxx * DERIVATIVE_STEP, atlas_position) );
}