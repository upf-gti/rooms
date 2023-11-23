#include mesh_includes.wgsl

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
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

// Uncharted 2 tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
fn tonemap_uncharted2_imp( color : vec3f ) -> vec3f
{
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

fn tonemap_uncharted( c : vec3f ) -> vec3f
{
    let W = 11.2;
    let color = tonemap_uncharted2_imp(c * 2.0);
    let whiteScale = 1.0 / tonemap_uncharted2_imp( vec3f(W) );
    return color * whiteScale;//LINEARtoSRGB(color * whiteScale);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    
    var view = normalize( in.world_position - camera_data.eye );

    var out: FragmentOutput;
    var color : vec3f = textureSample(albedo_texture, texture_sampler, view).rgb;

    color = pow(color, vec3f(1.0/2.2));

    // simple reinhard
    // color = color / (color + vec3f(1.0));
    color = tonemap_uncharted(color);

    out.color = vec4f(color, 1.0);

    return out;
}