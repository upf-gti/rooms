#include mesh_includes.wgsl
#include tonemappers.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(2) @binding(7) var sampler_clamp : sampler;


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    var world_position = instance_data.model * vec4f(in.position, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera_data.view_projection * world_position;
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * instance_data.color.rgb;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    
    var view = normalize( in.world_position - camera_data.eye );

    var out: FragmentOutput;
    var final_color : vec3f = textureSampleLevel(irradiance_texture, sampler_clamp, view, 0).rgb;
    
    // simple reinhard
    // color = color / (color + vec3f(1.0));
    final_color = tonemap_filmic(final_color);

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0);

    return out;
}