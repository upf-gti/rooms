//https://medium.com/@calebfaith/implementing-msdf-font-in-opengl-ea09a9ab7e00

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec3f,

};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
};

struct RenderMeshData {
    model  : mat4x4f,
    color  : vec3f,
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(0) var<uniform> mesh_data : RenderMeshData;
@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera_data.view_projection * mesh_data.model * vec4f(in.position, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * mesh_data.color;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

fn median( r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    var bgColor : vec4f = vec4f(0,0,0,0);
    var fgColor : vec4f = vec4f(in.color, 1.0);
    var pxRange : f32 = 4;

    var width : f32;
    var height : f32;
    // txColor.GetDimensions(width, height);

    var size : vec2f = vec2f(width,height);

    var uvs : vec2f = in.uv/size;
    var sample : vec3f = vec3f(1,1,1);//txColor.Sample(samLinear, uvs).xyz;

    size.x *= dpdx( uvs.x );
    size.y *= dpdy( uvs.y );

    var msdfUnit : vec2f = pxRange/size;
    var sigDist : f32 = median(sample.r, sample.g, sample.b) - 0.5;
    sigDist *= dot(msdfUnit, 0.5 / fwidth(uvs));
    var w : f32 = fwidth( sigDist );
    var opacity : f32 = smoothstep( 0.5 - w, 0.5 + w, sigDist );

    if(opacity < 0.01) {
        discard;
    }

    out.color = mix(bgColor, fgColor, opacity);

    return out;
}