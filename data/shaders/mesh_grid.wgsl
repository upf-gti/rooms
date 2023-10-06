struct VertexInput {
    @builtin(instance_index) instance_id : u32,
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
    color  : vec4f,
};

struct InstanceData {
    data : array<RenderMeshData>
}

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(0) var<storage, read> mesh_data : InstanceData;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : RenderMeshData = mesh_data.data[in.instance_id];

    var out: VertexOutput;
    out.position = camera_data.view_projection * instance_data.model * vec4f(in.position, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color * instance_data.color.rgb;
    out.normal = in.normal;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

const GRID_AREA_SIZE : f32 = 3.0;
const GRID_QUAD_SIZE :f32 = 0.25;
const LINE_WIDTH : f32 = 0.05;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    let wrapped_uvs : vec2f = fract(in.uv * GRID_AREA_SIZE / GRID_QUAD_SIZE);
    let line_width_proportion : f32 = GRID_QUAD_SIZE * LINE_WIDTH;
    let one_minus_line_width : f32 = 1.0 - line_width_proportion;
    let zero_plus_line_width : f32 = 0.0 + line_width_proportion;

    var out: FragmentOutput;

    if (wrapped_uvs.x > one_minus_line_width || wrapped_uvs.x < zero_plus_line_width) || (wrapped_uvs.y > one_minus_line_width || wrapped_uvs.y < zero_plus_line_width)
    {
        out.color = vec4f(0.57, 0.57, 0.57, 1.0);
    } else {
        discard;
    }

    //out.color = vec4f(wrapped_uvs.x, wrapped_uvs.y, 0.0, 1.0); // Color

    return out;
}