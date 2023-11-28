#include mesh_includes.wgsl
#include pbr_functions.wgsl

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var brdf_lut_texture: texture_2d<f32>;
@group(2) @binding(1) var brdf_lut_sampler : sampler;
@group(2) @binding(2) var irradiance_texture: texture_cube<f32>;
@group(2) @binding(3) var irradiance_sampler: sampler;

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

fn get_indirect_light( m : LitMaterial ) -> vec3f
{
    var cos_theta : f32 = max(dot(m.normal, m.view_dir), 0.0);

    // IBL
    // Specular + Diffuse

    // Specular color

    var F : vec3f = FresnelSchlickRoughness(cos_theta, m.specular_color, m.roughness);
    var k_s : vec3f = F;

    var mip_index : f32 = m.roughness * 5.0;
    var prefiltered_color : vec3f = textureSampleLevel(irradiance_texture, irradiance_sampler, m.reflected_dir, mip_index).rgb;
    //prefiltered_color = pow(prefiltered_color, vec3f(2.2));

    let brdf_coords : vec2f = vec2f(cos_theta, 1.0 - m.roughness);
    let brdf_lut : vec2f = textureSample(brdf_lut_texture, brdf_lut_sampler, brdf_coords).rg;

    var specular : vec3f = prefiltered_color * (F * brdf_lut.x + brdf_lut.y);

    // Diffuse sample: get last prefiltered mipmap
    var irradiance : vec3f = textureSampleLevel(irradiance_texture, irradiance_sampler, m.reflected_dir, 6).rgb;

    // Diffuse color
    var k_d : vec3f = 1.0 - k_s;
    var diffuse : vec3f = k_d * Diffuse(m.diffuse_color) * irradiance;

    // Combine factors and add AO
    return (diffuse + specular) * m.ao;
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
    m.metallic = 0.5;
    m.roughness = 0.5;
    m.diffuse_color = m.albedo * ( 1.0 - m.metallic );
    m.specular_color = mix(vec3f(0.04), m.albedo, m.metallic);
    m.ao = 1.0;
    m.view_dir = normalize(camera_data.eye - m.pos);
    m.reflected_dir = reflect( -m.view_dir, m.normal);

    // var distance : f32 = length(light_position - m.pos);
    // var attenuation : f32 = pow(1.0 - saturate(distance/light_max_radius), 1.5);
    var final_color : vec3f = vec3f(0.0); 
    final_color += get_direct_light(m, vec3f(1.0), 1.0);
    final_color += get_indirect_light(m);

    final_color = tonemap_uncharted(pow(final_color, vec3f(2.2)));

    out.color = vec4f(final_color, 1.0);

    return out;
}