
// Lights

struct LitMaterial
{
    pos : vec3f,
    normal : vec3f,
    albedo : vec3f,
    emissive : vec3f,
    diffuse_color : vec3f,
    specular_color : vec3f,
    metallic : f32,
    roughness : f32,
    ao : f32,
    view_dir : vec3f,
    reflected_dir : vec3f
};

fn get_direct_light( m : LitMaterial, shadow_factor : vec3f, attenuation : f32) -> vec3f
{
    var N : vec3f = normalize(m.normal);
    var V : vec3f = normalize(m.view_dir);

    var F0 : vec3f = m.specular_color;

    // hardcoded!!
    let light_position = vec3f(0.2, 0.5, 1.0);
    let light_color = vec3f(1.0);
    let light_intensity = 5.0;

    // calculate light radiance
    var L : vec3f = normalize(light_position - m.pos);
    var H : vec3f = normalize(V + L);
    var radiance : vec3f = light_color * light_intensity * attenuation * shadow_factor;

    // Cook-Torrance BRDF
    var NDF : f32 = DistributionGGX2(N, H, m.roughness);
    var G : f32 = GeometrySmith(N, V, L, m.roughness);
    var F : vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);

    var numerator : vec3f  = NDF * G * F;
    var denominator : f32 = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    var specular : vec3f = numerator / denominator;

    var k_s : vec3f = F;
    var k_d : vec3f = vec3f(1.0) - k_s;

    var NdotL : f32 = max(dot(N, L), 0.0);

    var final_color : vec3f = ((k_d * Diffuse(m.diffuse_color)) + specular) * radiance * NdotL;

    return final_color;
}

fn get_indirect_light( m : LitMaterial ) -> vec3f
{
    var cos_theta : f32 = max(dot(m.normal, m.view_dir), 0.0);

    // IBL
    // Specular + Diffuse

    // Specular color

    var F : vec3f = FresnelSchlickRoughness(cos_theta, m.specular_color, m.roughness);
    var k_s : vec3f = F;

    var mip_index : f32 = m.roughness * 6.0;
    var prefiltered_color : vec3f = textureSampleLevel(irradiance_texture, sampler_clamp, m.reflected_dir, mip_index).rgb;

    let brdf_coords : vec2f = vec2f(cos_theta, m.roughness);
    var brdf_lut : vec2f = textureSampleLevel(brdf_lut_texture, sampler_clamp, brdf_coords, 0).rg;

    var specular : vec3f = prefiltered_color * (F * brdf_lut.x + brdf_lut.y);

    // Diffuse sample: get last prefiltered mipmap
    var irradiance : vec3f = textureSampleLevel(irradiance_texture, sampler_clamp, m.normal, 6).rgb;

    // Diffuse color
    var k_d : vec3f = 1.0 - k_s;
    var diffuse : vec3f = k_d * (m.diffuse_color) * irradiance;

    // Combine factors and add AO
    return (diffuse) * m.ao;
}
