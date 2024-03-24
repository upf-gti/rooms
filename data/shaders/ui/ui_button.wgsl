#include ui_palette.wgsl
#include ../mesh_includes.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

#ifdef ALBEDO_TEXTURE
@group(2) @binding(0) var albedo_texture: texture_2d<f32>;
#endif

@group(2) @binding(1) var<uniform> albedo: vec4f;

#ifdef USE_SAMPLER
@group(2) @binding(7) var texture_sampler : sampler;
#endif

@group(3) @binding(0) var<uniform> ui_data : UIData;

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

    // Mask button shape
    var uvs = in.uv;
    var dist : f32 = distance(uvs, vec2f(0.5));
    var button_radius : f32 = 0.42;

    var out: FragmentOutput;

#ifdef ALBEDO_TEXTURE
    var color : vec4f = textureSample(albedo_texture, texture_sampler, in.uv);
#else
    var color : vec4f = vec4f(1.0);
#endif

    var selected_color = COLOR_HIGHLIGHT_DARK;
    var highlight_color = COLOR_SECONDARY;
    var back_color = vec3f(0.01);
    var gradient_factor = pow(uvs.y, 2.5);

    if(ui_data.is_button_disabled > 0.0) {
        back_color = pow(vec3f(0.41, 0.38, 0.44), vec3f(2.2));
    }

    // Assign basic color
    var lum = color.r * 0.3 + color.g * 0.59 + color.b * 0.11;
    var _color = vec3f( 1.0 - smoothstep(0.15, 0.4, lum) );
    _color = max(_color, back_color);

    var keep_colors = (ui_data.keep_rgb + ui_data.is_color_button) > 0.0;

    if(keep_colors) {
        _color = color.rgb * in.color.rgb;
        highlight_color = vec3f(1.0);
    }

    if( ui_data.is_selected > 0.0 ) {
        if( !keep_colors ) {
            var sel_color = mix( COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, pow(uvs.x + uvs.y, 3.0) );
            back_color = select( sel_color, COLOR_SECONDARY, ui_data.is_hovered > 0.0 );
            _color = smoothstep(vec3f(0.25), vec3f(0.45), color.rgb) * 0.5;
        }
    } 
    // not selected but hovered
    else if( ui_data.is_hovered > 0.0 ) {
        if( !keep_colors ) {
            highlight_color = mix( COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, gradient_factor );
        }
    }

    _color = (back_color + _color * highlight_color) * color.a;

    // Process selection
    var outline_color_selected = mix( COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, gradient_factor );
    _color = mix(outline_color_selected, _color, 1.0 - step(0.46 + (1.0 - ui_data.is_selected), dist));

    // // Process hover
    // var outline_intensity = 0.8;
    // var outline_mask = step(0.375 + (1.0 - ui_data.is_hovered), dist) * outline_intensity;
    // var outline_color = mix( COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, gradient_factor );
    // _color = mix(outline_color, _color, 1 - outline_mask);

    var shadow : f32 = smoothstep(button_radius, 0.5, dist);
    
    if(dist > button_radius) {
        _color = vec3f(0.01);
    }

    if (GAMMA_CORRECTION == 1) {
        _color = pow(_color, vec3f(1.0 / 2.2));
    }

    out.color = vec4f(_color,  1.0 - shadow);

    return out;
}