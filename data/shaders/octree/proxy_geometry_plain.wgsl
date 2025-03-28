#include math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include tonemappers.wgsl

// Add the generic SDF rendering functions
#include sdf_render_functions.wgsl

#define GAMMA_CORRECTION

struct VertexInput {
    @builtin(instance_index) instance_id : u32,
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) edit_range: vec2u,
    @location(1) vertex_in_world_space : vec3f,
    @location(2) vertex_in_sculpt_space : vec3f,
    @location(3) @interpolate(flat) brick_center_in_sculpt_space : vec3f,
    @location(4) @interpolate(flat) atlas_tile_coordinate : vec3f,
    @location(5) in_atlas_pos : vec3f,
    @location(6) @interpolate(flat) has_previews : u32,
    @location(7) @interpolate(flat) model_index : u32
};

@group(0) @binding(1) var<storage, read> preview_stroke : PreviewStroke;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(4) var texture_sampler : sampler;
@group(0) @binding(5) var<storage, read> brick_buffers: BrickBuffers_ReadOnly;
@group(0) @binding(8) var read_material_sdf: texture_3d<u32>;
@group(0) @binding(9) var<storage, read> sculpt_instance_data: array<SculptInstanceData>;

#dynamic @group(1) @binding(0) var<uniform> camera_data : CameraData;

//@group(2) @binding(0) var<storage, read> octree : Octree_NonAtomic;
@group(2) @binding(1) var<storage, read> brick_index_buffer : array<u32>;
@group(2) @binding(2) var<storage, read> sculpt_indirect : SculptIndirectCall_NonAtomic;

//@group(2) @binding(3) var<storage, read> ray_intersection_info: RayIntersectionInfo;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let sculpt_instance_count : u32 = sculpt_indirect.instance_count / sculpt_indirect.brick_count;
    let brick_idx : u32 = brick_index_buffer[in.instance_id / sculpt_instance_count];
    let model_idx : u32 = sculpt_indirect.starting_model_idx + (in.instance_id % sculpt_instance_count);

    let instance_data : ProxyInstanceData = brick_buffers.brick_instance_data[brick_idx];

    var vertex_in_sculpt_space : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var vertex_in_world_space : vec4f = (sculpt_instance_data[model_idx].model * vec4f(vertex_in_sculpt_space, 1.0));

    //let o = octree.current_level;

    var out: VertexOutput;
    out.position = camera_data.view_projection * vertex_in_world_space;

    out.edit_range = vec2u(instance_data.edit_id_start, instance_data.edit_count);

    out.has_previews = 0;
    if ((instance_data.in_use & BRICK_HAS_PREVIEW_FLAG) == BRICK_HAS_PREVIEW_FLAG) {
        out.has_previews = 1u;
    }
    
    out.vertex_in_sculpt_space = vertex_in_sculpt_space;
    out.brick_center_in_sculpt_space = instance_data.position;
    out.atlas_tile_coordinate = vec3f(u32(ATLAS_BRICK_SIZE) * vec3u(instance_data.atlas_tile_index % NUM_BRICKS_IN_ATLAS_AXIS,
                                                  (instance_data.atlas_tile_index / NUM_BRICKS_IN_ATLAS_AXIS) % NUM_BRICKS_IN_ATLAS_AXIS,
                                                   instance_data.atlas_tile_index / (NUM_BRICKS_IN_ATLAS_AXIS * NUM_BRICKS_IN_ATLAS_AXIS))) / SDF_RESOLUTION;
    out.vertex_in_world_space = vertex_in_world_space.xyz; 
    // From mesh space -1 to 1, -> 0 to 6.0/SDF_RESOLUTION (plus a voxel for padding)
    out.in_atlas_pos = (in.position * 0.5 + 0.5) * ATLAS_BRICK_NO_BORDER_SIZE/SDF_RESOLUTION + BRICK_VOXEL_ATLAS_SIZE + out.atlas_tile_coordinate;
    out.model_index = model_idx;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

#define MAX_LIGHTS

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;
@group(3) @binding(3) var<uniform> lights : array<Light, MAX_LIGHTS>;
@group(3) @binding(4) var<uniform> num_lights : u32;

fn sample_material_raw(pos : vec3u) -> SdfMaterial {
    let sample : u32 = textureLoad(read_material_sdf, pos, 0).r;

    return unpack_material(sample);
}

// From: http://paulbourke.net/miscellaneous/interpolation/
fn interpolate_material(pos : vec3f) -> SdfMaterial {
    var result : SdfMaterial;

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

fn sample_sdf_atlas(atlas_position : vec3f) -> f32
{
    return textureSampleLevel(read_sdf, texture_sampler, atlas_position, 0.0).r / SCULPT_MAX_SIZE;
}

fn sample_sdf_with_preview(sculpt_position : vec3f, atlas_position : vec3f) -> Surface
{
    var material : SdfMaterial;
    material.albedo = preview_stroke.stroke.material.color.xyz;
    material.roughness = preview_stroke.stroke.material.roughness;
    material.metallic = preview_stroke.stroke.material.metallic;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    surface.material = interpolate_material(atlas_position * SDF_RESOLUTION);
    
    surface = evaluate_stroke(sculpt_position, &(preview_stroke.stroke), &(preview_stroke.edit_list), surface, edit_range.x, edit_range.y);
    
    return surface;
}

fn sample_sdf_with_preview_without_material(sculpt_position : vec3f, atlas_position : vec3f) -> f32
{
    var material : SdfMaterial;

    var surface : Surface;
    surface.distance = sample_sdf_atlas(atlas_position);
    
    surface = evaluate_stroke(sculpt_position, &(preview_stroke.stroke), &(preview_stroke.edit_list), surface, edit_range.x, edit_range.y);

    
    return surface.distance;
}

fn sample_material_atlas(atlas_position : vec3f) -> SdfMaterial {
    return interpolate_material(atlas_position * SDF_RESOLUTION);
}

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
        let model_matrix = sculpt_instance_data[preview_stroke.current_sculpt_idx].model;
        let position_in_world : vec3f = (model_matrix * vec4f(position_in_sculpt, 1.0)).xyz;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(position_in_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_with_previews(position_in_sculpt, position_in_atlas);
        let normal_world : vec3f = normalize(adjoint(model_matrix) * normal);
        let ray_dir_world : vec3f = normalize(adjoint(model_matrix) * ray_dir);

        // let interpolant : f32 = (f32( i ) / f32(MAX_ITERATIONS)) * (M_PI / 2.0);
        // var heatmap_color : vec3f;
        // heatmap_color.r = sin(interpolant);
        // heatmap_color.g = sin(interpolant * 2.0);
        // heatmap_color.b = cos(interpolant);
        // return vec4f(heatmap_color, depth);
        //let material : SdfMaterial = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		return vec4f(apply_light(-ray_dir_world, position_in_world, position_in_world, normal_world, lightPos + lightOffset, surface.material), depth);
        //return vec4f(normal*0.5 + 0.50, depth);
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


fn raymarch(ray_origin_in_atlas_space : vec3f, ray_origin_in_sculpt_space : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f, min_hit_dist : f32) -> vec4f
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
        depth += distance * step(min_hit_dist, distance);
        depth = min(depth, max_distance);
        
        // position_in_atlas = ray_origin_in_atlas_space + ray_dir * depth;
        // distance = sample_sdf_atlas(position_in_atlas);
        // depth += distance * step(MIN_HIT_DIST, distance);
        // depth = min(depth, max_distance);

		if (distance < min_hit_dist) {
            exit = 1u;
            break;
		} 
	}

    if (exit == 1u ) {
        // From atlas position, to sculpt, to world
        let model_matrix : mat4x4f = sculpt_instance_data[model_index].model;
        let position_in_sculpt : vec3f = ray_origin_in_sculpt_space + ray_dir * (depth / SCULPT_TO_ATLAS_CONVERSION_FACTOR);
        let world_space_position : vec3f = (model_matrix * vec4f(position_in_sculpt, 1.0)).xyz;

        let curr_sculpt_flags : u32 = sculpt_instance_data[model_index].flags;
        let oof : bool = (curr_sculpt_flags & SCULPT_INSTANCE_IS_OUT_OF_FOCUS) == SCULPT_INSTANCE_IS_OUT_OF_FOCUS;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        var proj_pos : vec4f = view_proj * vec4f(world_space_position + ray_dir * epsilon, 1.0);
        // if(oof) {
        //     proj_pos.z -= 0.005;
        // }
        depth = (proj_pos.z) / proj_pos.w;

        let normal : vec3f = estimate_normal_atlas(position_in_atlas);
        let normal_world : vec3f = normalize(adjoint(model_matrix) * normal);
        let ray_dir_world : vec3f = normalize(adjoint(model_matrix) * ray_dir);

        let material : SdfMaterial = sample_material_atlas(position_in_atlas);

        var final_color : vec3f = vec3f(0.0);
        let v_dot_n : f32 = clamp(dot(-ray_dir_world, normal_world), 0.0, 1.0);

        if (oof) {
            let grey : f32 = 0.21 * material.albedo.r + 0.71 * material.albedo.g + 0.07 * material.albedo.b;
            final_color = vec3f(v_dot_n * grey) * 0.5;
        } else {
            final_color = apply_light(-ray_dir_world, world_space_position, world_space_position, normal_world, lightPos + lightOffset, material);

            var fresnel : f32 = pow(1.0 - v_dot_n, 3.0) + 0.5;
            fresnel = smoothstep(0.5, 1.0, fresnel);

            if ((curr_sculpt_flags & SCULPT_INSTANCE_IS_HOVERED) == SCULPT_INSTANCE_IS_HOVERED) {
                let hightlight_color : vec3f = mix(COLOR_PRIMARY, COLOR_SECONDARY, normal.y*0.5+0.5);
                final_color = mix(final_color, hightlight_color, fresnel);
            } else if ((curr_sculpt_flags & SCULPT_INSTANCE_IS_SELECTED) == SCULPT_INSTANCE_IS_SELECTED) {
                let hightlight_color : vec3f = mix(COLOR_TERCIARY, COLOR_HIGHLIGHT_LIGHT, 0.5);
                final_color = mix(final_color, hightlight_color, fresnel);
            }
        }

        // let interpolant : f32 = (f32( i ) / f32(MAX_ITERATIONS)) * (M_PI / 2.0);
        // var heatmap_color : vec3f;
        // heatmap_color.r = sin(interpolant);
        // heatmap_color.g = sin(interpolant * 2.0);
        // heatmap_color.b = cos(interpolant);
        // return vec4f(heatmap_color, depth);
        //let material : SdfMaterial = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
        return vec4f(final_color, depth);
        //return vec4f(normal_world*0.5 + 0.50, depth);
        //return vec4f(material.albedo, depth);
        //return vec4f(normal, depth);
        //return vec4f(vec3f(material.albedo), depth);
	}

    return vec4f(0.0, 0.0, 0.0, 0.999);
}

var<private> edit_range : vec2u;
var<private> model_index : u32;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    edit_range = in.edit_range;
    model_index = in.model_index;

    // TODO: move this to CPU!
    // From world to sculpt: make it relative to the sculpt center, and un-apply the rotation.
    let eye_sculpt_space = sculpt_instance_data[in.model_index].inv_model * vec4f(camera_data.eye, 1.0);

    // Get the sculpt space position relative to the current brick (brick-space)
    let eye_brick_space : vec3f = eye_sculpt_space.xyz - in.brick_center_in_sculpt_space;
    // Atlas and sculpt space are aligned, the only difference is a change of scale, depending on brick size. Now the coordinates are Atlas-brick relative
    var eye_atlas_pos : vec3f = eye_brick_space * SCULPT_TO_ATLAS_CONVERSION_FACTOR;
    // make the coordinate accurate to the "global" in-brick position
    eye_atlas_pos += in.atlas_tile_coordinate + vec3f(BRICK_ATLAS_HALF_SIZE);
    let ray_dir_atlas : vec3f = normalize(in.in_atlas_pos - eye_atlas_pos);


    // let ray_dir_world : vec3f = normalize(in.vertex_in_world_space - eye_atlas_pos);

    // watchouttt
    let ray_dir_sculpt : vec3f = normalize(in.vertex_in_sculpt_space.xyz - eye_sculpt_space.xyz);
    // let ray_dir_world : vec3f = normalize(adjoint(sculpt_instance_data[in.model_index].model) * ray_dir_sculpt);

    // let eye_obj_distance : f32 = abs(length(in.vertex_in_sculpt_space.xyz - eye_sculpt_space.xyz));

    // Max raymarch distances
    let raymarch_distance : f32 = ray_intersect_AABB_only_near(in.in_atlas_pos, -ray_dir_atlas, in.atlas_tile_coordinate + vec3f(BRICK_ATLAS_HALF_SIZE), vec3f(BRICK_NO_BORDER_ATLAS_SIZE));
    let raymarch_distance_sculpt_space : f32 = ray_intersect_AABB_only_near(in.vertex_in_sculpt_space.xyz, -ray_dir_sculpt, in.brick_center_in_sculpt_space, vec3f(BRICK_WORLD_SIZE));

    // Get proper ray origins as we render brick with back-faces
    let ray_origin : vec3f = in.in_atlas_pos.xyz + raymarch_distance * (-ray_dir_atlas);
    let ray_origin_sculpt_space : vec3f = in.vertex_in_sculpt_space.xyz + raymarch_distance_sculpt_space * (-ray_dir_sculpt);

    var ray_result : vec4f;
    // let tmp = preview_stroke.stroke.material.color.x;

    let curr_sculpt_flags : u32 = sculpt_instance_data[model_index].flags;
    let oof : bool = (curr_sculpt_flags & SCULPT_INSTANCE_IS_OUT_OF_FOCUS) == SCULPT_INSTANCE_IS_OUT_OF_FOCUS;

    if (in.has_previews == 1 && !oof) {
        ray_result = raymarch_with_previews(ray_origin, ray_origin_sculpt_space, ray_dir_atlas, raymarch_distance, camera_data.view_projection);
    } else {
        let curr_min_hit_dist : f32 = MIN_HIT_DIST; // Basic raymarch LODding: mix(MIN_HIT_DIST, MAX_HIT_DIST, max((eye_obj_distance / 2.0), 0.0));
        ray_result = raymarch(ray_origin, ray_origin_sculpt_space, ray_dir_atlas, raymarch_distance, camera_data.view_projection, curr_min_hit_dist);
    }
    var final_color : vec3f = ray_result.rgb; 
    

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    // if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
    //     if (in.has_previews == 1u) {
    //         out.color = vec4f(0.0, 1.0, 0.0, 1.0);
    //     } else {
    //         out.color = vec4f(0.0, 0.0, 0.0, 1.0);
    //     }
        
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