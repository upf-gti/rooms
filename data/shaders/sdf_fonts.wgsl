//https://medium.com/@calebfaith/implementing-msdf-font-in-opengl-ea09a9ab7e00

#include mesh_includes.wgsl

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(0) var texture: texture_2d<f32>;
@group(2) @binding(4) var texture_sampler : sampler;

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

fn median( r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn screenPxRange( pxRange : f32, texCoord : vec2f ) -> f32 {
    var unitRange = vec2f(pxRange) / vec2f(textureDimensions(texture));
    var screenTexSize = vec2f(1.0) / fwidth(texCoord);
    return max(0.5*dot(unitRange, screenTexSize), 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    
    var dummy = camera_data.eye;

    var out: FragmentOutput;

    var bgColor : vec4f = vec4f(0.0, 0.0, 0.0, 0.0);
    var fgColor : vec4f = vec4f(in.color, 1.0);

    var sz : vec2f = vec2f(textureDimensions(texture));
    // var msd : vec3f = textureLoad(texture, vec2u(in.uv * sz), 0).rgb;
    var msd : vec3f = textureSample(texture, texture_sampler, in.uv).rgb;
    var sd : f32 = median(msd.r, msd.g, msd.b);
    var screenPxDistance = screenPxRange(4.0, in.uv) * (sd - 0.5);
    var opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    if (opacity < 0.01) {
        discard;
    }
    
    out.color = mix(bgColor, fgColor, opacity);
    return out;
}