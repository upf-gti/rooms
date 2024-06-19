#include math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include tonemappers.wgsl

#define GAMMA_CORRECTION

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
    @location(1) @interpolate(flat) edit_range: vec2u,
    @location(3) vertex_in_world_space : vec3f,
    @location(4) vertex_in_sculpt_space : vec3f,
    @location(5) @interpolate(flat) brick_center_in_sculpt_space : vec3f,
    @location(6) @interpolate(flat) atlas_tile_coordinate : vec3f,
    @location(7) in_atlas_pos : vec3f,
    @location(8) @interpolate(flat) has_previews : u32,
    @location(9) @interpolate(flat) model_index : u32
};

struct CameraData {
    view_projection : mat4x4f,
    eye_position : vec3f,
    dummy : f32
};

@group(0) @binding(0) var<storage, read> brick_copy_buffer : array<u32>;
@group(0) @binding(1) var<storage, read> preview_stroke : PreviewStroke;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(4) var texture_sampler : sampler;
@group(0) @binding(5) var<storage, read> brick_buffers: BrickBuffers_ReadOnly;
@group(0) @binding(8) var read_material_sdf: texture_3d<u32>;
@group(0) @binding(9) var<storage, read> sculpt_model_buffer: array<mat4x4f>;

#dynamic @group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(3) var<storage, read> ray_intersection_info: RayIntersectionInfo;

// From https://gist.github.com/rsms/9d9e7c4eadf9fe23da0bf0bfb96bc2e6
fn inverse(m :mat4x4f) -> mat4x4f {
        // Note: wgsl does not have an inverse() (matrix inverse) function built in.
        // Source adapted from https://github.com/glslify/glsl-inverse/blob/master/index.glsl
        let a00 = m[0][0];
        let a01 = m[0][1];
        let a02 = m[0][2];
        let a03 = m[0][3];
        let a10 = m[1][0];
        let a11 = m[1][1];
        let a12 = m[1][2];
        let a13 = m[1][3];
        let a20 = m[2][0];
        let a21 = m[2][1];
        let a22 = m[2][2];
        let a23 = m[2][3];
        let a30 = m[3][0];
        let a31 = m[3][1];
        let a32 = m[3][2];
        let a33 = m[3][3];
        let b00 = a00 * a11 - a01 * a10;
        let b01 = a00 * a12 - a02 * a10;
        let b02 = a00 * a13 - a03 * a10;
        let b03 = a01 * a12 - a02 * a11;
        let b04 = a01 * a13 - a03 * a11;
        let b05 = a02 * a13 - a03 * a12;
        let b06 = a20 * a31 - a21 * a30;
        let b07 = a20 * a32 - a22 * a30;
        let b08 = a20 * a33 - a23 * a30;
        let b09 = a21 * a32 - a22 * a31;
        let b10 = a21 * a33 - a23 * a31;
        let b11 = a22 * a33 - a23 * a32;
        let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        return mat4x4f(
          vec4<f32>(
            a11 * b11 - a12 * b10 + a13 * b09,
            a02 * b10 - a01 * b11 - a03 * b09,
            a31 * b05 - a32 * b04 + a33 * b03,
            a22 * b04 - a21 * b05 - a23 * b03),
          vec4<f32>(
            a12 * b08 - a10 * b11 - a13 * b07,
            a00 * b11 - a02 * b08 + a03 * b07,
            a32 * b02 - a30 * b05 - a33 * b01,
            a20 * b05 - a22 * b02 + a23 * b01),
          vec4<f32>(
            a10 * b10 - a11 * b08 + a13 * b06,
            a01 * b08 - a00 * b10 - a03 * b06,
            a30 * b04 - a31 * b02 + a33 * b00,
            a21 * b02 - a20 * b04 - a23 * b00),
          vec4<f32>(
            a11 * b07 - a10 * b09 - a12 * b06,
            a00 * b09 - a01 * b07 + a02 * b06,
            a31 * b01 - a30 * b03 - a32 * b00,
            a20 * b03 - a21 * b01 + a22 * b00)
        ) * (1.0 / det);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let raw_instance_data : u32 = brick_copy_buffer[in.instance_id];

    let instance_index : u32 = raw_instance_data >> 20u;
    let model_index : u32 = raw_instance_data & 0xFFF;

    let instance_data : ProxyInstanceData = brick_buffers.brick_instance_data[instance_index];

    let tile_pointer : u32 = ray_intersection_info.tile_pointer;

    var vertex_in_sculpt_space : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var vertex_in_world_space : vec3f = rotate_point_quat(vertex_in_sculpt_space, sculpt_data.sculpt_rotation);
    vertex_in_world_space += sculpt_data.sculpt_start_position;

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    // vertex_in_world_space = vec4f(rotate_point_quat(vertex_in_world_space.xyz, sculpt_data.sculpt_rotation), 1.0);
    out.position = camera_data.view_projection * (sculpt_model_buffer[model_index] * vec4f(vertex_in_world_space, 1.0));
    out.uv = in.uv; // forward to the fragment shader

    out.edit_range = vec2u(instance_data.edit_id_start, instance_data.edit_count);

    out.has_previews = 0;
    if ((instance_data.in_use & BRICK_HAS_PREVIEW_FLAG) == BRICK_HAS_PREVIEW_FLAG) {
        out.has_previews = 1u;
    }
    
    out.vertex_in_sculpt_space = vertex_in_sculpt_space;
    out.brick_center_in_sculpt_space = instance_data.position;
    // This is in an attribute for debugging
    out.atlas_tile_coordinate = vec3f(10 * vec3u(instance_data.atlas_tile_index % BRICK_COUNT,
                                                  (instance_data.atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   instance_data.atlas_tile_index / (BRICK_COUNT * BRICK_COUNT))) / SDF_RESOLUTION;
    out.vertex_in_world_space = vertex_in_world_space.xyz; 
    // From mesh space -1 to 1, -> 0 to 8.0/SDF_RESOLUTION (plus a voxel for padding)
    out.in_atlas_pos = (in.position * 0.5 + 0.5) * 8.0/SDF_RESOLUTION + 1.0/SDF_RESOLUTION + out.atlas_tile_coordinate;
    out.model_index = model_index;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

@group(2) @binding(0) var<uniform> sculpt_data : SculptData;

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;
@group(3) @binding(3) var<uniform> lights : array<Light, MAX_LIGHTS>;
@group(3) @binding(4) var<uniform> num_lights : u32;

fn sample_material_raw(pos : vec3u) -> Material {
    let sample : u32 = textureLoad(read_material_sdf, pos, 0).r;

    return unpack_material(sample);
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

fn sample_material_atlas(atlas_position : vec3f) -> Material {
    return interpolate_material(atlas_position * SDF_RESOLUTION);
}

fn sample_sdf_atlas(atlas_position : vec3f) -> f32
{
    return textureSampleLevel(read_sdf, texture_sampler, atlas_position, 0.0).r / SCULPT_MAX_SIZE;
}

fn sample_sdf_with_preview(sculpt_position : vec3f, atlas_position : vec3f) -> Surface
{
    var material : Material;
    material.albedo = preview_stroke.stroke.material.color.xyz;
    material.roughness = preview_stroke.stroke.material.roughness;
    material.metalness = preview_stroke.stroke.material.metallic;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    surface.material = interpolate_material(atlas_position * SDF_RESOLUTION);
    
    surface = evaluate_stroke(sculpt_position, &(preview_stroke.stroke), &(preview_stroke.edit_list), surface, edit_range.x, edit_range.y);
    
    return surface;
}

fn sample_sdf_with_preview_without_material(sculpt_position : vec3f, atlas_position : vec3f) -> f32
{
    var material : Material;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    
    surface = evaluate_stroke(sculpt_position, &(preview_stroke.stroke), &(preview_stroke.edit_list), surface, edit_range.x, edit_range.y);

    
    return surface.distance;
}

// Add the generic SDF rendering functions
#include sdf_render_functions.wgsl

// https://iquilezles.org/articles/normalsSDF/
fn estimate_normal_with_previews( p : vec3f, p_world: vec3f) -> vec3f
{
    let k : vec2f = vec2f(1.0, -1.0);
    return normalize( k.xyy * sample_sdf_with_preview_without_material( p + k.xyy * DERIVATIVE_STEP, p_world + k.xyy * DERIVATIVE_STEP) + 
                      k.yyx * sample_sdf_with_preview_without_material( p + k.yyx * DERIVATIVE_STEP, p_world + k.yyx * DERIVATIVE_STEP) + 
                      k.yxy * sample_sdf_with_preview_without_material( p + k.yxy * DERIVATIVE_STEP, p_world + k.yxy * DERIVATIVE_STEP) + 
                      k.xxx * sample_sdf_with_preview_without_material( p + k.xxx * DERIVATIVE_STEP, p_world + k.xxx * DERIVATIVE_STEP) );
}

fn raymarch_with_previews(ray_origin_atlas_space : vec3f, ray_origin_sculpt_space : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var distance : f32;

    var surface : Surface;
    var position_in_atlas : vec3f;
    var position_in_sculpt : vec3f;
    var i : i32 = 0;
    var exit : u32 = 0u;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		position_in_atlas = ray_origin_atlas_space + ray_dir * depth;
        position_in_sculpt = ray_origin_sculpt_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        surface = sample_sdf_with_preview(position_in_sculpt, position_in_atlas);
        distance = surface.distance;
        depth += distance * step(MIN_HIT_DIST, distance);
        depth = min(depth, max_distance);
        
        // position_in_atlas = ray_origin_in_atlas_space + ray_dir * depth;
        // distance = sample_sdf_atlas(position_in_atlas);
        // depth += distance * step(MIN_HIT_DIST, distance);
        // depth = min(depth, max_distance);

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 
	}

    if (exit == 1u ) {
        // From atlas position, to sculpt, to world
        //position_in_sculpt = ray_origin_atlas_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        //position_in_atlas = ray_origin_atlas_space + ray_dir * depth;
        let position_in_world : vec3f = rotate_point_quat(position_in_sculpt, (sculpt_data.sculpt_rotation)) + sculpt_data.sculpt_start_position;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(position_in_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_with_previews(position_in_sculpt, position_in_atlas);

        // let interpolant : f32 = (f32( i ) / f32(MAX_ITERATIONS)) * (M_PI / 2.0);
        // var heatmap_color : vec3f;
        // heatmap_color.r = sin(interpolant);
        // heatmap_color.g = sin(interpolant * 2.0);
        // heatmap_color.b = cos(interpolant);
        // return vec4f(heatmap_color, depth);
        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		//return vec4f(apply_light(-ray_dir, position_in_world, position_in_world, normal, lightPos + lightOffset, surface.material), depth);
        return vec4f(normal*0.5 + 0.50, depth);
        //return vec4f(material.albedo, depth);
        //return vec4f(normal, depth);
        //return vec4f(vec3f(material.albedo), depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(0.0, 0.0, 0.0, 0.999);
}

// https://iquilezles.org/articles/normalsSDF/
fn estimate_normal_atlas(atlas_position : vec3f) -> vec3f
{
    let k : vec2f = vec2f(1.0, -1.0);
    return normalize( k.xyy * sample_sdf_atlas(atlas_position + k.xyy * DERIVATIVE_STEP) + 
                      k.yyx * sample_sdf_atlas(atlas_position + k.yyx * DERIVATIVE_STEP) + 
                      k.yxy * sample_sdf_atlas(atlas_position + k.yxy * DERIVATIVE_STEP) + 
                      k.xxx * sample_sdf_atlas(atlas_position + k.xxx * DERIVATIVE_STEP) );
}


fn raymarch(ray_origin_in_atlas_space : vec3f, ray_origin_in_sculpt_space : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var distance : f32;

    var position_in_atlas : vec3f;
    var i : i32 = 0;
    var exit : u32 = 0u;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		position_in_atlas = ray_origin_in_atlas_space + ray_dir * depth;
        distance = sample_sdf_atlas(position_in_atlas);
        depth += distance * step(MIN_HIT_DIST, distance);
        depth = min(depth, max_distance);
        
        // position_in_atlas = ray_origin_in_atlas_space + ray_dir * depth;
        // distance = sample_sdf_atlas(position_in_atlas);
        // depth += distance * step(MIN_HIT_DIST, distance);
        // depth = min(depth, max_distance);

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 
	}

    if (exit == 1u ) {
        // From atlas position, to sculpt, to world
        let position_in_sculpt : vec3f = ray_origin_in_sculpt_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        let position_in_world : vec3f = rotate_point_quat(position_in_sculpt, (sculpt_data.sculpt_rotation)) + sculpt_data.sculpt_start_position;
        let world_space_position : vec3f = (sculpt_model_buffer[model_index] * vec4f(position_in_world, 1.0)).xyz;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(world_space_position + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_atlas(position_in_atlas);

        let material : Material = sample_material_atlas(position_in_atlas);
        // let interpolant : f32 = (f32( i ) / f32(MAX_ITERATIONS)) * (M_PI / 2.0);
        // var heatmap_color : vec3f;
        // heatmap_color.r = sin(interpolant);
        // heatmap_color.g = sin(interpolant * 2.0);
        // heatmap_color.b = cos(interpolant);
        // return vec4f(heatmap_color, depth);
        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		return vec4f(apply_light(-ray_dir, world_space_position, world_space_position, normal, lightPos + lightOffset, material), depth);
        //return vec4f(normal*0.5 + 0.50, depth);
        //return vec4f(material.albedo, depth);
        //return vec4f(normal, depth);
        //return vec4f(vec3f(material.albedo), depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(0.0, 0.0, 0.0, 0.999);
}

var<private> edit_range : vec2u;
var<private> model_index : u32;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    edit_range = in.edit_range;
    model_index = in.model_index;

    let eye_in_world_sculpt = inverse(sculpt_model_buffer[in.model_index]) * vec4f(camera_data.eye_position, 1.0);

    // From world to sculpt: make it relative to the sculpt center, and un-apply the rotation.
    var eye_atlas_pos : vec3f = rotate_point_quat(eye_in_world_sculpt.xyz - sculpt_data.sculpt_start_position, quat_conj(sculpt_data.sculpt_rotation));
    // Get the sculpt space position relative to the current brick
    eye_atlas_pos -= in.brick_center_in_sculpt_space;
    // Atlas and sculpt space are aligned, the only difference is a change of scale, depednign on brick size. Now the coordinates are Atlas-brick relative
    eye_atlas_pos *= SCULPT_TO_ATLAS_CONVERSION_FACTOR;
    // make the coordinate accurate to the "global" in-brick position
    eye_atlas_pos += in.atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION); 
    let ray_dir_atlas : vec3f = normalize(in.in_atlas_pos - eye_atlas_pos);

    let ray_dir_world : vec3f = normalize(in.vertex_in_world_space.xyz - eye_in_world_sculpt.xyz);
    let ray_dir_sculpt : vec3f = rotate_point_quat(ray_dir_world, quat_conj(sculpt_data.sculpt_rotation));
    // ray dir in atlas coords :((

    let raymarch_distance_sculpt_space : f32 = ray_intersect_AABB_only_near(in.vertex_in_sculpt_space.xyz, -ray_dir_sculpt, in.brick_center_in_sculpt_space, vec3f(BRICK_WORLD_SIZE));
    let ray_origin_sculpt_space : vec3f = in.vertex_in_sculpt_space.xyz + raymarch_distance_sculpt_space * (-ray_dir_sculpt);
    
    // Raro
    let raymarch_distance : f32 = ray_intersect_AABB_only_near(in.in_atlas_pos, -ray_dir_atlas, in.atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION), vec3f(BRICK_ATLAS_SIZE));

    let ray_origin : vec3f = in.in_atlas_pos.xyz + raymarch_distance * (-ray_dir_atlas);

    var ray_result : vec4f;
    if (in.has_previews == 1) {
        ray_result = raymarch_with_previews(ray_origin, ray_origin_sculpt_space, ray_dir_atlas, raymarch_distance, camera_data.view_projection);
    } else {
        ray_result = raymarch(ray_origin, ray_origin_sculpt_space, ray_dir_atlas, raymarch_distance, camera_data.view_projection);
    }
    var final_color : vec3f = ray_result.rgb; 

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
        if (in.has_previews == 1u) {
            out.color = vec4f(0.0, 1.0, 0.0, 1.0);
        } else {
            out.color = vec4f(0.0, 0.0, 0.0, 1.0);
        }
        
        out.depth = in.position.z;
    }

    // out.color = vec4f(raymarch_distance / (SQRT_3 * BRICK_WORLD_SIZE), 0.0, 0.0, 0.0); // Color
    // out.depth = in.position.z / in.position.w;

    // out.color = vec4f(1.0, 0.0, 0.0, 1.0); // Color
    // out.depth = 0.0;

    if (out.depth == 0.999) {
        //let tmp : vec4f =  textureSampleLevel(read_material_sdf, texture_sampler, in.position.xyz, 0.0);
        //let sample : vec4<u32> = textureLoad(read_material_sdf, vec3<u32>(in.position.xyz));
        discard;
    }

    return out;
}