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
    @location(3) tangent: vec3f,
    @location(4) color: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) vertex_in_world_space : vec3f,
    @location(4) vertex_in_sculpt_space : vec3f,
    @location(5) @interpolate(flat) brick_center_in_sculpt_space : vec3f,
    @location(6) @interpolate(flat) atlas_tile_coordinate : vec3f,
    @location(7) in_atlas_pos : vec3f,
    @location(8) @interpolate(flat) has_previews : u32
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(0) var<storage, read> brick_copy_buffer : array<u32>;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(4) var texture_sampler : sampler;
@group(0) @binding(5) var<storage, read> octree_proxy_data: OctreeProxyInstancesNonAtomic;
@group(0) @binding(8) var read_material_sdf: texture_3d<u32>;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@group(2) @binding(3) var<storage, read> ray_intersection_info: RayIntersectionInfo;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_index : u32 = brick_copy_buffer[in.instance_id];
    let instance_data : ProxyInstanceData = octree_proxy_data.instance_data[instance_index];

    let tile_pointer : u32 = ray_intersection_info.tile_pointer;

    var vertex_in_sculpt_space : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var vertex_in_world_space : vec3f = rotate_point_quat(vertex_in_sculpt_space, sculpt_data.sculpt_rotation);
    vertex_in_world_space += sculpt_data.sculpt_start_position;

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    // vertex_in_world_space = vec4f(rotate_point_quat(vertex_in_world_space.xyz, sculpt_data.sculpt_rotation), 1.0);
    out.position = camera_data.view_projection * vec4f(vertex_in_world_space, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = vec3f(0.0, 0.0, 0.0);
    out.normal = in.normal;

    out.has_previews = 0;
    if ((instance_data.in_use & BRICK_HAS_PREVIEW_FLAG) == BRICK_HAS_PREVIEW_FLAG) {
        out.color = vec3f(0.0, 1.0, 0.0);
        out.has_previews = 1u;
    }

    if ((tile_pointer & 0x3FFFFFFFu) == instance_index) {
        out.color = vec3f(0.0, 0.0, 1.0);
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

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

@group(0) @binding(1) var<uniform> eye_position : vec3f;

@group(2) @binding(0) var<uniform> sculpt_data : SculptData;
@group(2) @binding(1) var<storage, read> preview_data : PreviewDataReadonly;

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;
@group(3) @binding(3) var<uniform> lights : array<Light, MAX_LIGHTS>;

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
    material.albedo = preview_data.preview_stroke.material.color.xyz;
    material.roughness = preview_data.preview_stroke.material.roughness;
    material.metalness = preview_data.preview_stroke.material.metallic;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    surface.material = interpolate_material(atlas_position * SDF_RESOLUTION);
    
    for(var i : u32 = 0u; i < preview_data.preview_stroke.edit_count; i++) {
        surface = evaluate_edit(sculpt_position, preview_data.preview_stroke.primitive, preview_data.preview_stroke.operation, preview_data.preview_stroke.parameters, preview_data.preview_stroke.color_blend_op, surface, material, preview_data.preview_stroke.edits[i]);
        surface.distance = surface.distance / 2.0; 
    }
    
    return surface;
}

fn sample_sdf_with_preview_without_material(sculpt_position : vec3f, atlas_position : vec3f) -> f32
{
    var material : Material;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    
    for(var i : u32 = 0u; i < preview_data.preview_stroke.edit_count; i++) {
        surface = evaluate_edit(sculpt_position, preview_data.preview_stroke.primitive, preview_data.preview_stroke.operation, preview_data.preview_stroke.parameters, preview_data.preview_stroke.color_blend_op, surface, material, preview_data.preview_stroke.edits[i]);
    }
    
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
    var surface : Surface;
    var distance : f32;

    var pos_sculpt_space : vec3f;
    var pos_atlas_space : vec3f;
    var i : i32 = 0;
    var exit : u32 = 0u;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		pos_sculpt_space = ray_origin_sculpt_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        pos_atlas_space = ray_origin_atlas_space + ray_dir * depth ;

        surface = sample_sdf_with_preview(pos_sculpt_space, pos_atlas_space);
        distance = surface.distance;

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 

        depth += distance;
	}

    if (exit == 1u) {
        
        let pos_world : vec3f = rotate_point_quat(pos_sculpt_space, (sculpt_data.sculpt_rotation)) + sculpt_data.sculpt_start_position;
        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(pos_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_with_previews(pos_sculpt_space, pos_atlas_space);

        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		return vec4f(apply_light(-ray_dir, pos_sculpt_space, pos_world, normal, lightPos + lightOffset, surface.material), depth);
        //return vec4f(normal, depth);
        //return vec4f(surface.material.albedo, depth);
        //return vec4f(vec3f(surface.material.albedo), depth);
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

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 

        depth += distance;
	}

    if (exit == 1u) {
        // From atlas position, to sculpt, to world
        let position_in_sculpt : vec3f = ray_origin_in_sculpt_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        let position_in_world : vec3f = rotate_point_quat(position_in_sculpt, (sculpt_data.sculpt_rotation)) + sculpt_data.sculpt_start_position;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(position_in_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_atlas(position_in_atlas);

        last_found_surface_distance = distance;

        let material : Material = sample_material_atlas(position_in_atlas);
        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		return vec4f(apply_light(-ray_dir, position_in_world, position_in_world, normal, lightPos + lightOffset, material), depth);
        //return vec4f(normal, depth);
        //return vec4f(material.albedo, depth);
        //return vec4f(normal, depth);
        //return vec4f(vec3f(material.albedo), depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(0.0, 0.0, 0.0, 0.999);
}

var<private> brick_center_in_sculpt_space : vec3f;
var<private> atlas_tile_coordinates : vec3f;

var<private> last_found_surface_distance : f32;

var<private> last_sampled_material : Material;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    // From world to sculpt: make it relative to the sculpt center, and un-apply the rotation.
    var eye_atlas_pos : vec3f = rotate_point_quat(eye_position - sculpt_data.sculpt_start_position, quat_conj(sculpt_data.sculpt_rotation));
    // Get the sculpt space position relative to the current brick
    eye_atlas_pos -= in.brick_center_in_sculpt_space;
    // Atlas and sculpt space are aligned, the only difference is a change of scale, depednign on brick size. Now the coordinates are Atlas-brick relative
    eye_atlas_pos *= SCULPT_TO_ATLAS_CONVERSION_FACTOR;
    // make the coordinate accurate to the "global" in-brick position
    eye_atlas_pos += in.atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION); 
    let ray_dir_atlas : vec3f = normalize(in.in_atlas_pos - eye_atlas_pos);

    let ray_dir_world : vec3f = normalize(in.vertex_in_world_space.xyz - eye_position);
    let ray_dir_sculpt : vec3f = rotate_point_quat(ray_dir_world, quat_conj(sculpt_data.sculpt_rotation));
    // ray dir in atlas coords :((
    
    // Raro
    let raymarch_distance : f32 = ray_intersect_AABB_only_near(in.in_atlas_pos, ray_dir_atlas, in.atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION), vec3f(BRICK_ATLAS_SIZE));
    
    atlas_tile_coordinates = in.atlas_tile_coordinate;
    brick_center_in_sculpt_space = in.brick_center_in_sculpt_space;

    var ray_result : vec4f;
    if (in.has_previews == 1) {
        ray_result = raymarch_with_previews(in.in_atlas_pos.xyz, in.vertex_in_sculpt_space.xyz, ray_dir_atlas, raymarch_distance, camera_data.view_projection);
    } else {
        ray_result = raymarch(in.in_atlas_pos.xyz, in.vertex_in_sculpt_space.xyz, ray_dir_atlas, raymarch_distance, camera_data.view_projection);
    }
    var final_color : vec3f = ray_result.rgb; 

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    // if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
    //     out.color = vec4f(in.color.x, in.color.y, in.color.z, 1.0);
    //     out.depth = in.position.z;
    // }

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