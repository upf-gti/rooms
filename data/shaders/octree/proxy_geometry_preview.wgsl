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
    @location(5) @interpolate(flat) voxel_center_world : vec3f
};

struct CameraData {
    view_projection : mat4x4f,
};

@group(0) @binding(1) var<uniform> eye_position : vec3f;

@group(1) @binding(0) var<uniform> camera_data : CameraData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let instance_data : ProxyInstanceData = preview_data.instance_data[in.instance_id];

    var voxel_pos : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var world_pos : vec3f = rotate_point_quat(voxel_pos, sculpt_data.sculpt_rotation);
    world_pos += sculpt_data.sculpt_start_position;

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    // world_pos = vec4f(rotate_point_quat(world_pos.xyz, sculpt_data.sculpt_rotation), 1.0);
    out.position = camera_data.view_projection * vec4f(world_pos, 1.0);
    out.uv = in.uv; // forward to the fragment shader
    out.color = in.color;
    out.normal = in.normal;
    
    out.voxel_pos = voxel_pos;
    out.voxel_center_world = rotate_point_quat(instance_data.position, sculpt_data.sculpt_rotation) + sculpt_data.sculpt_start_position;
    // This is in an attribute for debugging
    out.world_pos = world_pos.xyz; 
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}


@group(2) @binding(0) var<uniform> sculpt_data : SculptData;
@group(2) @binding(1) var<storage, read> preview_data : PreviewDataReadonly;

@group(3) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(3) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(3) @binding(2) var sampler_clamp: sampler;



fn sample_material(pos : vec3f, padding : vec3f) -> Material {
    var material : Material;
    material.albedo = preview_data.preview_stroke.color.xyz;
    material.roughness = preview_data.preview_stroke.material.x;
    material.metalness = preview_data.preview_stroke.material.y;
    return material;
}


fn sample_sdf(position : vec3f, padding : vec3f) -> f32
{
    // TODO: preview edits
    var material : Material = sample_material(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0));
    var surface : Surface;
    surface.distance = 10000.0;
    for(var i : u32 = 0u; i < preview_data.preview_stroke.edit_count; i++) {
        surface = evaluate_edit(position, preview_data.preview_stroke.primitive, preview_data.preview_stroke.operation, preview_data.preview_stroke.parameters, surface, material, preview_data.preview_stroke.edits[i]);
    }
    return surface.distance;
}

var<private> last_found_surface_distance : f32;

// Add the generic SDF rendering functions
#include sdf_render_functions.wgsl

fn raymarch_world(ray_origin_world : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var distance : f32;

    var pos : vec3f;
    var i : i32 = 0;
    var exit : u32 = 0u;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		pos = ray_origin_world + ray_dir * depth;

        distance = sample_sdf(pos, vec3f(0.0));

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 

        depth += distance;
	}

    if (exit == 1u) {
        let pos_world : vec3f = pos + sculpt_data.sculpt_start_position;
        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(pos_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal(pos, vec3f(0.0));

        let material : Material = sample_material(pos, vec3f(0.0, 0.0, 0.0));
        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
		return vec4f(apply_light(-ray_dir, pos, pos_world, lightPos + lightOffset, material), depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(vec3f(0.0), 0.999);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    let ray_dir : vec3f = normalize(in.world_pos.xyz - eye_position);

    let raymarch_distance : f32 = ray_AABB_intersection_distance(in.world_pos.xyz, ray_dir, in.voxel_center_world, vec3f(BRICK_WORLD_SIZE));

    let ray_result = raymarch_world(in.world_pos.xyz - sculpt_data.sculpt_start_position, ray_dir, raymarch_distance, camera_data.view_projection);

    var final_color : vec3f = ray_result.rgb; 

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    // if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
    //     out.color = vec4f(0.0, 0.0, 1.0, 1.0);
    //     out.depth = in.position.z;
    // } else {
    //     //discard;
    // }

    // out.color = vec4f(1.0, 0.0, 0.0, 1.0); // Color
    // out.depth = 0.0;

    if (out.depth == 0.999) {
        //let tmp : vec4f =  textureSampleLevel(read_material_sdf, texture_sampler, in.position.xyz, 0.0);
        //let sample : vec4<u32> = textureLoad(read_material_sdf, vec3<u32>(in.position.xyz));
        discard;
    }

    return out;
}