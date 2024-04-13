#include ../pbr_functions.wgsl
#include ../pbr_light.wgsl

#include sdf_utils.wgsl

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
    m.f90 = vec3f(1.0);
    m.specular_weight = 1.0;

    m.ao = 1.0;

    var final_color : vec3f = vec3f(0.0);
    final_color += get_indirect_light(m);
    final_color += get_direct_light(m);
    final_color += m.emissive;

    return tonemap_khronos_pbr_neutral(final_color);
}