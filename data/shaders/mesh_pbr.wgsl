#include mesh_includes.wgsl
#include pbr_functions.wgsl
#include tonemappers.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

#ifdef ALBEDO_TEXTURE
@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
#else
@group(2) @binding(0) var<uniform> albedo: vec4f;
#endif

#ifdef METALLIC_ROUGHNESS_TEXTURE
@group(2) @binding(1) var metallic_roughness_texture: texture_2d<f32>;
#else
@group(2) @binding(1) var<uniform> metallic_roughness: vec2f;
#endif

#ifdef NORMAL_TEXTURE
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
#endif

#ifdef EMISSIVE_TEXTURE
@group(2) @binding(3) var emissive_texture: texture_2d<f32>;
#else
@group(2) @binding(3) var<uniform> emissive: vec3f;
#endif

#ifdef USE_SAMPLER
@group(2) @binding(4) var sampler_2d : sampler;
#endif

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    var dummy = camera_data.eye;

    var m : LitMaterial;

    m.pos = in.world_position;

    // Vectors

    m.normal = normalize(in.normal);
    m.view_dir = normalize(camera_data.eye - m.pos);

#ifdef NORMAL_TEXTURE
    var normal_color = textureSample(normal_texture, sampler_2d, in.uv).rgb;
    m.normal = perturb_normal(m.normal, m.view_dir, in.uv, normal_color);
#endif

    m.reflected_dir = reflect( -m.view_dir, m.normal);

    // Material properties

#ifdef ALBEDO_TEXTURE
    m.albedo = textureSample(albedo_texture, sampler_2d, in.uv).rgb * in.color;
    m.albedo = pow(m.albedo, vec3f(2.2));
#else
    m.albedo = albedo.rgb;
#endif

#ifdef EMISSIVE_TEXTURE
    m.emissive = textureSample(emissive_texture, sampler_2d, in.uv).rgb;
    m.emissive = pow(m.emissive, vec3f(2.2));
#else
    m.emissive = emissive;
#endif

#ifdef METALLIC_ROUGHNESS_TEXTURE
    var metal_rough : vec3f = textureSample(metallic_roughness_texture, sampler_2d, in.uv).rgb;
    m.metallic = metal_rough.b;
    m.roughness = max(metal_rough.g, 0.04);
#else
    m.metallic = metallic_roughness.x;
    m.roughness = metallic_roughness.y;
#endif

    m.diffuse_color = m.albedo * ( 1.0 - m.metallic );
    m.specular_color = mix(vec3f(0.04), m.albedo, m.metallic);
    m.ao = 1.0;

    // var distance : f32 = length(light_position - m.pos);
    // var attenuation : f32 = pow(1.0 - saturate(distance/light_max_radius), 1.5);
    var final_color : vec3f = vec3f(0.0); 
    // final_color += get_direct_light(m, vec3f(1.0), 1.0);

    final_color += tonemap_uncharted(get_indirect_light(m));

    final_color += m.emissive;

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0);

    return out;
}