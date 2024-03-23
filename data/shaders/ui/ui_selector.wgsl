#include ../mesh_includes.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var<uniform> ui_data : UIData;

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

    var dummy = camera_data.eye;
    var dummy1 = ui_data.is_color_button;

    var out: FragmentOutput;

    var uvs = in.uv;
    var dist : f32 = distance(uvs, vec2f(0.5));
    var button_radius : f32 = 0.375;
    var shadow : f32 = smoothstep(button_radius, 0.43, dist);

    let back_color = in.color;
    var final_color : vec3f = back_color;

    if(dist > button_radius) {
        final_color *= 0.5;
    }
    else {
        shadow = smoothstep(0.3, 0.1, dist);
    }

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3f(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0 - shadow);
    
    return out;
}