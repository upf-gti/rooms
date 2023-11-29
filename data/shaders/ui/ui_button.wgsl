#include ../mesh_includes.wgsl

#define GAMMA_CORRECTION

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
    color = pow(color, vec4f(2.2));

    // Mask button shape
    var uvs = in.uv;
    var d : f32 = distance(uvs, vec2f(0.5));
    var alpha_mask = 1.0 - step(0.5, d);
    if( alpha_mask < 0.1 ) {
        discard;
    }

    let selected_color = vec3f(0.15, 0.02, 0.9);
    let hover_color = vec3f(0.87, 0.6, 0.02);
    var back_color = vec3f(0.01);

    // Assign basic color
    var lum = color.r  * 0.3 + color.g * 0.59 + color.b * 0.11;
    var _color = vec3f( 1.0 - smoothstep(0.15, 0.4, lum) ) * hover_color;
    _color = max(_color, back_color);

    var keep_colors = (ui_data.keep_rgb + ui_data.is_color_button) > 0.0;

    if(keep_colors) {
        _color = color.rgb * in.color;
    }

    if( ui_data.is_selected > 0.0 && !keep_colors ) {
        back_color = hover_color - pow(d, 1.75);
        _color = pow(smoothstep(vec3f(0.3), vec3f(0.7), color.rgb), vec3f(1.2));
    }

    _color = select( back_color, _color, color.a > 0.3 );

    // Process selection
    var outline_color_selected = mix( selected_color, hover_color, uvs.x * uvs.y );
    _color = mix(outline_color_selected, _color, 1 - step(0.46 + (1.0 - ui_data.is_selected), d));  

    // Process hover
    var outline_intensity = 0.8;
    var outline_mask = step(0.46 + (1.0 - ui_data.is_hovered), d) * outline_intensity;
    var outline_color = mix( selected_color, hover_color, uvs.y );
    _color = mix(outline_color, _color, 1 - outline_mask);

    if (GAMMA_CORRECTION == 1) {
        _color = pow(_color, vec3f(1.0 / 2.2));
    }

    out.color = vec4f(_color, alpha_mask);

    return out;
}