#include ../mesh_includes.wgsl

#define GAMMA_CORRECTION

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(1) var<uniform> albedo: vec4f;

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

// from @lingel at https://www.shadertoy.com/view/3tj3R1

const EPSILON : f32 = 0.01;
const PI : f32 = 3.14159265359;

fn hsv2rgb( hsv : vec3f ) -> vec3f
{
    var c : vec3f = hsv;
    c.x /= 360.0;
    var K : vec4f = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    var p : vec3f = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx,vec3f(0.0),vec3f(1.0)), c.y);
}

fn get_ring_color( uvs : vec2f ) -> vec3f
{
    let xiangxian : f32 = floor(1.0 - uvs.y );
    let radian : f32 = xiangxian * 2.0 * PI + atan2(uvs.y,uvs.x);
    let degree : f32 = radian/(2.0*PI)*360.0;
    let hsv = vec3(degree,1.0,1.0);
    return hsv2rgb(hsv);
}

fn position2sv(p : vec2f, v1 : vec2f, v2 : vec2f, v3 : vec2f) -> vec2f
{
    var baseS : vec2f = v1 - v3;
    var baseV : vec2f = baseS * 0.5 - (v2 - v3);
    var baseO : vec2f = v3 - baseV;
    var pp : vec2f = p - baseO;
    var s : f32 = dot(pp, baseS) / pow(length(baseS), 2.0);
    var v : f32 = dot(pp, baseV) / pow(length(baseV), 2.0);
    s -= 0.5;
    s /= v;
    s += 0.5;
    return clamp(vec2f(s, v), vec2f(0.0), vec2f(1.0));
}

fn draw_triangle( uv : vec2f, v1 : vec2f, v2 : vec2f, v3 : vec2f, H : f32 ) -> vec3f
{
    return hsv2rgb(vec3f(H, position2sv(uv,v1,v2,v3)));
}

fn calculate_triangle_weight(p : vec2f, v1 : vec2f, v2 : vec2f, v3 : vec2f) -> vec3f
{
    var weight : vec3f;
    weight.x = ((v2.y-v3.y)*(p.x-v3.x)+(v3.x-v2.x)*(p.y-v3.y)) / ((v2.y-v3.y)*(v1.x-v3.x)+(v3.x-v2.x)*(v1.y-v3.y));
    weight.y = ((v3.y-v1.y)*(p.x-v3.x)+(v1.x-v3.x)*(p.y-v3.y)) / ((v2.y-v3.y)*(v1.x-v3.x)+(v3.x-v2.x)*(v1.y-v3.y));
    weight.z = 1.0 - weight.x - weight.y;
    return weight;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var dummy = camera_data.eye;

    // Mask button shape
    var dist : f32 = distance(in.uv, vec2f(0.5));
    var button_radius : f32 = 0.44;

    var out: FragmentOutput;

    var uvs : vec2f = in.uv * 2.0 - 1.0;
    uvs.y *= -1.0;

    var current_color = ui_data.picker_color.rgb * ui_data.picker_color.a;
    var final_color : vec3f = current_color;

    // Ring alpha
    let exterior_radius : f32 = 1.0;
    let thickness : f32 = 0.3;
    let interior_radius : f32 = exterior_radius - thickness;

    let position_radius : f32 = length(uvs);
    let interior_mask : f32 = smoothstep(interior_radius, interior_radius + EPSILON, position_radius);
    let outerior_mask : f32 = smoothstep(exterior_radius, exterior_radius - EPSILON, position_radius);
    var alpha = min(interior_mask, outerior_mask);  

    final_color = mix(final_color, get_ring_color(uvs), alpha);

    var v1 : vec2f = vec2f(0.0);
    var v2 : vec2f = vec2f(0.0);
    var v3 : vec2f = vec2f(0.0);

    let degree : f32 = 90.0;

    // Compute points
    let r : f32 = PI / 180.0;
    var radian : f32 = degree * r;
    v1.x = cos(radian) * interior_radius;
    v1.y = sin(radian) * interior_radius;
    radian += 120.0 * r;
    v2.x = cos(radian) * interior_radius;
    v2.y = sin(radian) * interior_radius;
    radian += 120.0 * r;
    v3.x = cos(radian) * interior_radius;
    v3.y = sin(radian) * interior_radius;

    // Triangle alpha
    var weight : vec3f = calculate_triangle_weight(uvs, v1, v2, v3);

    alpha = min(min(weight.x, weight.y), weight.z);
    alpha = smoothstep(-EPSILON * 0.5, EPSILON * 0.5, alpha);

    final_color = mix(final_color, draw_triangle(uvs, v1, v2, v3, degree), alpha);

    final_color = pow(final_color, vec3f(2.2));

    if(dist > button_radius) {
        final_color = vec3f(0.1);
    }

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3f(1.0 / 2.2));
    }

    var shadow : f32 = smoothstep(button_radius, 0.5, dist);

    out.color = vec4f(final_color, 1.0 - shadow);

    return out;
}