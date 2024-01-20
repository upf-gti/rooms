
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
    @location(3) world_position: vec3f,
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
    eye : vec3f,
    dummy : f32
};

struct UIData {
    is_hovered : f32,
    num_group_items : f32,
    is_selected : f32,
    is_color_button : f32,
    picker_color: vec4f,
    keep_rgb : f32,
    slider_value : f32,
    slider_max: f32,
    is_button_disabled : f32
};