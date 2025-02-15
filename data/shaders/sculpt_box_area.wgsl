#include mesh_includes.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

#dynamic @group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
@group(2) @binding(1) var<uniform> albedo: vec4f;
@group(2) @binding(7) var texture_sampler : sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    var world_position = instance_data.model * vec4f(in.position, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera_data.view_projection * world_position;
    out.uv = in.uv; // forward to the fragment shader
    out.color = vec4(in.color, 1.0) * albedo;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    
    var dummy = camera_data.eye;

    var out: FragmentOutput;
    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv * 0.75);

    var final_color = in.color.rgb;

    // use both controllers to check distance??...
    // let left_distance : f32 = length(camera_data.left_controller_position - in.world_position);
    let right_distance : f32 = length(camera_data.right_controller_position - in.world_position);
    let distance : f32 = right_distance; //min(left_distance, right_distance);

    var factor : f32 = 1.0 - smoothstep(0.15, 0.3, distance);

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3f(1.0 / 2.2));
    }

    out.color = vec4f(mix(vec3f(color.a), final_color, factor), max(color.a * factor, 0.025));

    return out;
}