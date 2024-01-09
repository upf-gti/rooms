#include ../math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include ../tonemappers.wgsl

#define GAMMA_CORRECTION

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
    @location(3) world_pos : vec3f,
    @location(4) voxel_pos : vec3f,
    @location(5) @interpolate(flat) voxel_center : vec3f,
    @location(6) @interpolate(flat) atlas_tile_coordinate : vec3f,
    @location(7) in_atlas_pos : vec3f
};

struct SculptData {
    sculpt_start_position   : vec3f,
    dummy1                  : f32,
    sculpt_rotation         : vec4f,
    sculpt_inv_rotation     : vec4f
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(0) var<storage, read> brick_copy_buffer : array<u32>;
@group(0) @binding(2) var texture_sampler : sampler;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> octree_proxy_data: OctreeProxyInstancesNonAtomic;
@group(0) @binding(8) var read_material_sdf: texture_3d<u32>;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_index : u32 = brick_copy_buffer[in.instance_id];
    let instance_data : ProxyInstanceData = octree_proxy_data.instance_data[instance_index];

    var voxel_pos : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var world_pos : vec3f = rotate_point_quat(voxel_pos, sculpt_data.sculpt_rotation);
    world_pos += sculpt_data.sculpt_start_position;

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    // world_pos = vec4f(rotate_point_quat(world_pos.xyz, sculpt_data.sculpt_rotation), 1.0);
    out.position = camera_data.view_projection * vec4f(world_pos, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = vec3f(0.0, 0.0, 0.0);
    out.normal = in.normal;

    if ((instance_data.in_use & BRICK_HAS_PREVIEW_FLAG) == BRICK_HAS_PREVIEW_FLAG) {
        out.color = vec3f(0.0, 1.0, 0.0);
    }
    
    out.voxel_pos = voxel_pos;
    out.voxel_center = instance_data.position;
    // This is in an attribute for debugging
    out.atlas_tile_coordinate = vec3f(10 * vec3u(instance_data.atlas_tile_index % BRICK_COUNT,
                                                  (instance_data.atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   instance_data.atlas_tile_index / (BRICK_COUNT * BRICK_COUNT))) / SDF_RESOLUTION;
    out.world_pos = world_pos.xyz; 
    // From mesh space -1 to 1, -> 0 to 8.0/SDF_RESOLUTION (plus a voxel for padding)
    out.in_atlas_pos = (in.position * 0.5 + 0.5) * 8.0/SDF_RESOLUTION + 1.0/SDF_RESOLUTION + out.atlas_tile_coordinate;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

@group(0) @binding(1) var<uniform> eye_position : vec3f;

@group(2) @binding(0) var<uniform> sculpt_data : SculptData;

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;


fn sample_material_raw(pos : vec3u) -> Material {
    let sample : u32 = textureLoad(read_material_sdf, pos, 0).r;

    return unpack_material(sample);
}

fn sample_sdf(position : vec3f) -> f32
{
    // TODO: preview edits
    return textureSampleLevel(read_sdf, texture_sampler, position, 0.0).r;
}

// From: http://paulbourke.net/miscellaneous/interpolation/
fn interpolate_material(pos : vec3f) -> Material {
    var result : Material;

    let pos_f_part : vec3f = abs(fract(pos));
    let pos_i_part : vec3u = vec3u(floor(pos));

    let index000 : vec3u = pos_i_part;
    let index010 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 1, pos_i_part.z + 0)  ;
    let index100 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 0, pos_i_part.z + 0)  ;
    let index001 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 0, pos_i_part.z + 1)  ;
    let index101 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 0, pos_i_part.z + 1)  ;
    let index011 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 1, pos_i_part.z + 1)  ;
    let index110 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 1, pos_i_part.z + 0)  ;
    let index111 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 1, pos_i_part.z + 1)  ;

    result = Material_mult_by(sample_material_raw(index000), (1.0 - pos_f_part.x) * (1.0 - pos_f_part.y) * (1.0 - pos_f_part.z));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index100), (pos_f_part.x) * (1.0 - pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index010), (1.0 - pos_f_part.x) * (pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index001), (1.0 - pos_f_part.x) * (1.0 - pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index101), (pos_f_part.x) * (1.0 - pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index011), (1.0 - pos_f_part.x) * (pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index110), (pos_f_part.x) * (pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_material_raw(index111), (pos_f_part.x) * (pos_f_part.y) * (pos_f_part.z)));

    return result;
}

fn sample_material(pos : vec3f) -> Material {
    return interpolate_material(pos * SDF_RESOLUTION);
}

// Add the generic SDF rendering functions
#include sdf_render_functions.wgsl

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    let ray_dir : vec3f = normalize(in.world_pos.xyz - eye_position);
    let ray_dir_voxel_space : vec3f = normalize(in.voxel_pos - rotate_point_quat(eye_position - sculpt_data.sculpt_start_position, sculpt_data.sculpt_inv_rotation));

    let raymarch_distance : f32 = ray_AABB_intersection_distance(in.voxel_pos, ray_dir_voxel_space, in.voxel_center, vec3f(BRICK_WORLD_SIZE));

    let ray_result = raymarch(in.in_atlas_pos.xyz, in.world_pos.xyz, ray_dir_voxel_space, raymarch_distance * SCALE_CONVERSION_FACTOR, camera_data.view_projection);

    var final_color : vec3f = ray_result.rgb; 

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
        out.color = vec4f(in.color.x, in.color.y, in.color.z, 1.0);
        out.depth = in.position.z;
    }

    // out.color = vec4f(1.0, 0.0, 0.0, 1.0); // Color
    // out.depth = 0.0;

    if (out.depth == 0.999) {
        //let tmp : vec4f =  textureSampleLevel(read_material_sdf, texture_sampler, in.position.xyz, 0.0);
        //let sample : vec4<u32> = textureLoad(read_material_sdf, vec3<u32>(in.position.xyz));
        discard;
    }

    return out;
}