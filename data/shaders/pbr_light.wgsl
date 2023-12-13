
// Lights

struct LitMaterial
{
    pos : vec3f,
    normal : vec3f,
    albedo : vec3f,
    emissive : vec3f,
    f0 : vec3f,
    c_diff : vec3f,
    metallic : f32,
    roughness : f32,
    ao : f32,
    view_dir : vec3f,
    reflected_dir : vec3f
};

// fn get_direct_light( m : LitMaterial, shadow_factor : vec3f, attenuation : f32) -> vec3f
// {
//     var N : vec3f = normalize(m.normal);
//     var V : vec3f = normalize(m.view_dir);

//     var F0 : vec3f = m.specular_color;

//     // hardcoded!!
//     let light_position = vec3f(0.2, 0.5, 1.0);
//     let light_color = vec3f(1.0);
//     let light_intensity = 5.0;

//     // calculate light radiance
//     var L : vec3f = normalize(light_position - m.pos);
//     var H : vec3f = normalize(V + L);
//     var radiance : vec3f = light_color * light_intensity * attenuation * shadow_factor;

//     // Cook-Torrance BRDF
//     var NDF : f32 = DistributionGGX2(N, H, m.roughness);
//     var G : f32 = GeometrySmith(N, V, L, m.roughness);
//     var F : vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);

//     var numerator : vec3f  = NDF * G * F;
//     var denominator : f32 = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
//     var specular : vec3f = numerator / denominator;

//     var k_s : vec3f = F;
//     var k_d : vec3f = vec3f(1.0) - k_s;

//     var NdotL : f32 = max(dot(N, L), 0.0);

//     var final_color : vec3f = ((k_d * Diffuse(m.diffuse_color)) + specular) * radiance * NdotL;

//     return final_color;
// }

fn get_indirect_light( m : LitMaterial ) -> vec3f
{
    let n_dot_v : f32 = clamp(dot(m.normal, m.view_dir), 0.0, 1.0);

    let max_mipmap : f32 = 5.0;

    let lod : f32 = m.roughness * max_mipmap;

    // IBL
    // Specular + Diffuse

    // Specular color

    let brdf_coords : vec2f = clamp(vec2f(n_dot_v, m.roughness), vec2f(0.0, 0.0), vec2f(1.0, 1.0));
    let brdf_lut : vec2f = textureSampleLevel(brdf_lut_texture, sampler_clamp, brdf_coords, 0).rg;
    
    let specular_sample : vec3f = textureSampleLevel(irradiance_texture, sampler_clamp, m.reflected_dir, lod).rgb;

    let k_s : vec3f = FresnelSchlickRoughness(n_dot_v, m.f0, m.roughness);
    let fss_ess : vec3f = (k_s * brdf_lut.x + brdf_lut.y);

    let specular : vec3f = specular_sample * fss_ess;

    // Diffuse sample: get last prefiltered mipmap
    let irradiance : vec3f = textureSampleLevel(irradiance_texture, sampler_clamp, m.normal, max_mipmap).rgb;

    // Diffuse color
    let ems : f32 = (1.0 - (brdf_lut.x + brdf_lut.y));
    let f_avg : vec3f = (m.f0 + (1.0 - m.f0) / 21.0);
    let fms_ems : vec3f = ems * fss_ess * f_avg / (1.0 - f_avg * ems);
    let diffuse : vec3f = m.c_diff * (1.0 - fss_ess + fms_ems);

    // Combine factors and add AO
    return (diffuse + specular) * m.ao;
}
