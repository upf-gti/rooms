#include mesh_includes.wgsl
#include pbr_functions.wgsl

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var albedo_texture: texture_cube<f32>;
@group(2) @binding(1) var texture_sampler : sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    var world_position = instance_data.model * vec4f(in.position, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera_data.view_projection * world_position;
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * instance_data.color.rgb;
    out.normal = (instance_data.model * vec4f(in.normal, 0.0)).xyz;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

fn get_indirect_light( m : LitMaterial ) -> vec4f
{
    var cos_theta : f32 = max(dot(m.normal, m.view_dir), 0.0);

    // IBL
    // Specular + Diffuse

    // Specular color

    var F : vec3f = FresnelSchlickRoughness(cos_theta, m.specular_color, m.roughness);
    var k_s : vec3f = F;

    var mip_index : f32 = m.roughness * 6.0;
    var prefiltered_color : vec3f = textureSampleLevel(albedo_texture, texture_sampler, m.reflected_dir, mip_index).rgb;
    prefiltered_color = pow(prefiltered_color, vec3f(1.0/2.2));

    let brdf_coords : vec2f = vec2f(cos_theta, 1.0 - m.roughness);
    let brdf_lut : vec2f = vec2f(0.0);//txBrdf_lUT.Sample(clampLinear, vec2f(cos_theta, 1.0 - m.roughness)).xy;

    var specular : vec3f = prefiltered_color * (F * brdf_lut.x + brdf_lut.y);

    // Diffuse sample: get last prefiltered mipmap
    var irradiance : vec3f = textureSampleLevel(albedo_texture, texture_sampler, m.reflected_dir, 6).rgb;

    // Diffuse color
    var k_d : vec3f = 1.0 - k_s;
    var diffuse : vec3f = k_d * m.diffuse_color * irradiance;

    // Combine factors and add AO
    var ibl : vec3f = (diffuse + specular) * m.ao;

    return vec4f(ibl, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    var dummy = camera_data.eye;

    var m : LitMaterial;

    m.pos = in.world_position;
    m.normal = normalize(in.normal);
    m.albedo = in.color;
    m.emissive = vec3f(0.0);
    m.metallic = 0.7;
    m.roughness = 0.4;
    m.diffuse_color = m.albedo * ( 1.0 - m.metallic );
    m.specular_color = mix(vec3f(0.04), m.albedo, m.metallic);
    m.ao = 1.0;
    m.view_dir = normalize(camera_data.eye - m.pos);
    m.reflected_dir = reflect( -m.view_dir, m.normal);

    // var distance : f32 = length(light_position - m.pos);
    // var attenuation : f32 = pow(1.0 - saturate(distance/light_max_radius), 1.5);
    var final_color : vec4f = get_direct_light( m, vec3f(1.0), 1.0 );
    final_color += get_indirect_light(m);

    // out.color = vec4f(pow(final_color.rgb, vec3f(2.2)), 1.0);
    out.color = vec4f(pow(final_color.rgb, vec3f(2.2)), 1.0);

    return out;
}