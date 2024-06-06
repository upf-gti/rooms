struct UIData {
    hover_info: vec2f,
    dummy: vec2f,
    xr_info: vec4f,
    aspect_ratio : f32,
    num_group_items : f32,
    is_selected : f32,
    is_color_button : f32,
    picker_color: vec3f,
    keep_rgb : f32,
    slider_value : f32,
    slider_max: f32,
    slider_min: f32,
    is_button_disabled : f32
};

const COLOR_PRIMARY         = pow(vec3f(0.976, 0.976, 0.976), vec3f(2.2));
const COLOR_SECONDARY       = pow(vec3f(0.967, 0.892, 0.793), vec3f(2.2));
const COLOR_TERCIARY        = pow(vec3f(1.0, 0.404, 0.0), vec3f(2.2));
const COLOR_HIGHLIGHT_LIGHT = pow(vec3f(0.467, 0.333, 0.933), vec3f(2.2));
const COLOR_HIGHLIGHT       = pow(vec3f(0.26, 0.2, 0.533), vec3f(2.2));
const COLOR_HIGHLIGHT_DARK  = pow(vec3f(0.082, 0.086, 0.196), vec3f(2.2));
const COLOR_DARK            = pow(vec3f(0.172, 0.172, 0.172), vec3f(2.2));

const EPSILON : f32 = 0.02;
const PI : f32 = 3.14159265359;
const OUTLINECOLOR : vec4f = vec4f(0.3, 0.3, 0.3, 0.9);

fn remap_range(oldValue : f32, oldMin: f32, oldMax : f32, newMin : f32, newMax : f32) -> f32
{
    return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
}

fn sdRoundedBox( p : vec2f, b : vec2f, cr : vec4f ) -> f32
{
    var r : vec2f = select(cr.zw, cr.xy, p.x > 0.0);
    r.x = select(r.y, r.x, p.y > 0.0);
    var q : vec2f = abs(p) - b + r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,vec2f(0.0))) - r.x;
}

fn calculate_triangle_weight( p : vec2f, v1 : vec2f, v2 : vec2f, v3 : vec2f ) -> vec3f
{
    var weight : vec3f;
    weight.x = ((v2.y-v3.y)*(p.x-v3.x)+(v3.x-v2.x)*(p.y-v3.y)) / ((v2.y-v3.y)*(v1.x-v3.x)+(v3.x-v2.x)*(v1.y-v3.y));
    weight.y = ((v3.y-v1.y)*(p.x-v3.x)+(v1.x-v3.x)*(p.y-v3.y)) / ((v2.y-v3.y)*(v1.x-v3.x)+(v3.x-v2.x)*(v1.y-v3.y));
    weight.z = 1.0 - weight.x - weight.y;
    return weight;
}

fn draw_line( uv : vec2f, p1 : vec2f, p2 : vec2f, color : vec4f, thickness : f32 ) -> vec4f
{
    var final_color : vec4f = color;
    let t = thickness * 0.5;
    var dir = p2 - p1;
    let line_length : f32 = length(dir);
    dir = normalize(dir);
    let to_uv : vec2f = uv - p1;
    let project_length : f32 = dot(dir, to_uv);
    var p2_line_distance : f32 = length(to_uv-dir*project_length);
    p2_line_distance = smoothstep(t + EPSILON, t, p2_line_distance);
    var p2_end_distance : f32 = select(project_length-line_length, abs(project_length), project_length <= 0.0);
    p2_end_distance = smoothstep(t, t - EPSILON * 0.5, p2_end_distance);
    final_color = vec4f(final_color.xyz * mix(p2_line_distance, 0.0, 0.01), final_color.a);
    final_color.a *= min(p2_line_distance, p2_end_distance);
    return final_color;
}

fn draw_point( uv : vec2f, p : vec2f, s : f32) -> vec4f
{
    let alpha : f32 = smoothstep(0.015,0.002, abs(length(uv - p) - s));
    return vec4(vec3(1.0), alpha);
}


struct VertexInput {
    @builtin(instance_index) instance_id : u32,
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) tangent: vec3f,
    @location(4) color: vec3f,
    @location(5) weights: vec4f,
    @location(6) joints: vec4i
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) world_position: vec3f,
};

struct RenderMeshData {
    model  : mat4x4f
};

struct InstanceData {
    data : array<RenderMeshData>
}

struct CameraData {
    view_projection : mat4x4f,
    eye : vec3f,
    dummy : f32,
    right_controller_position : vec3f,
    dummy2 : f32
};

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

#dynamic @group(1) @binding(0) var<uniform> camera_data : CameraData;

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

    let curvature : f32 = 0.25;

    let uv_centered : vec2f = in.uv * vec2f(2.0) - vec2f(1.0);
    let curve_factor : f32 = 1.0 - (abs(uv_centered.x * uv_centered.x) * 0.5 + 0.5);
    var curved_pos : vec3f = in.position;
    curved_pos.z -= curvature * curve_factor;

    var world_position : vec4f = instance_data.model * vec4f(curved_pos, 1.0);
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

    let is_hovered : bool = ui_data.hover_info.x > 0.0;
    let is_selected : bool = ui_data.is_selected > 0.0;

    var out: FragmentOutput;

    let global_scale : f32 = 0.95;
    let size : vec2f = ui_data.xr_info.xy;
    let is_button : bool = size.x != 1.0;
    
    var position : vec2f = ui_data.xr_info.zw;

#ifdef ALBEDO_TEXTURE
    var corrected_uv : vec2f = vec2f(in.uv.x, 1.0 - in.uv.y);
    corrected_uv = corrected_uv / size;
    corrected_uv = corrected_uv - (position / size) + 0.5;
    // corrected_uv = corrected_uv + (size * 0.5);
    corrected_uv.y = 1.0 - corrected_uv.y;
    var color : vec4f = textureSample(albedo_texture, texture_sampler, corrected_uv);
#else
    var color : vec4f = vec4f(ui_data.picker_color, 1.0);
#endif

    var final_color = select(color.rgb, COLOR_SECONDARY, is_button);

    // center stuff
    position -= vec2f(0.5);
    
    var ra : vec4f = vec4f(0.125);
    var si : vec2f = vec2f(ui_data.aspect_ratio, 1.0) * size * global_scale;
    ra = min(ra, min(vec4f(si.x), vec4f(si.y)));
    var uvs = vec2f(in.uv.x, 1.0 - in.uv.y) - position;
    var pos : vec2f = vec2(uvs * 2.0 - 1.0);
    pos.x *= ui_data.aspect_ratio;

    let d : f32 = sdRoundedBox(pos, si, ra);

    if(is_hovered) {
        final_color = mix(COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, pow(corrected_uv.y, 2.0));
    }

    if(is_selected) {
        final_color = COLOR_PRIMARY;
    }

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3f(1.0 / 2.2));
    }

    var alpha : f32 = (1.0 - smoothstep(0.0, 0.04, d)) * color.a;

    if(is_button) {
        alpha += (1.0 - smoothstep(0.0, 0.015, abs(d)));
        final_color = mix( final_color, vec3f(0.5), 1.0 - smoothstep(0.0, 0.015, abs(d)) );
    }

    out.color = vec4f(final_color, alpha);

    return out;
}