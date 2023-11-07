#include ../mesh_includes.wgsl

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
@group(2) @binding(1) var texture_sampler : sampler;

@group(3) @binding(0) var<uniform> ui_data : UIData;

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

    var out: FragmentOutput;

    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv);

    if (color.a < 0.01) {
        discard;
    }
    
    let hover_color = vec3f(0.95, 0.76, 0.17);
    var _color = color.rgb;

    if( ui_data.is_color_button > 0.0 )  {
        _color *= in.color;
    }

    let selected_color = vec3f(0.47, 0.37, 0.94);
    var widget_color = mix(in.color, hover_color, ui_data.is_hovered);
    var mask = distance(in.uv, vec2f(0.5));

    if( ui_data.is_selected > 0.0 ) {

        var icon_mask = smoothstep(1 - color.r, 0.5, 1.0);
        _color = mix(vec3f(1 - icon_mask) * (1 - mask), selected_color, icon_mask);
        _color = max(_color, vec3f(0.12));
    } 

    var masked_color : vec3f;

    if( ui_data.is_color_button > 0.0 ) {
        mask = step(0.45, mask);
        var border_color = mix(widget_color, vec3f(0.2,0.2,0.2), ui_data.is_selected);
        masked_color = mix(in.color, mix(border_color, hover_color, ui_data.is_hovered), mask);
    } else {
        mask = step(0.45 + (1.0 - ui_data.is_hovered), mask);
        masked_color = mix(widget_color, _color, 1.0 - mask);
    }
   
    out.color = vec4f(pow(masked_color, vec3f(2.2)), color.a);

    return out;
}