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
    @location(1) @interpolate(flat) is_interior: u32,
    @location(2) vertex_in_world_space : vec3f,
    @location(3) vertex_in_sculpt_space : vec3f,
    @location(4) @interpolate(flat) voxel_center_sculpt_space : vec3f,
};

#dynamic @group(0) @binding(0) var<uniform> camera_data : CameraData;

// @group(1) @binding(0) var<uniform> sculpt_data : SculptData;
@group(1) @binding(1) var<storage, read> preview_stroke : PreviewStroke;
@group(1) @binding(5) var<storage, read> brick_buffers: BrickBuffers_ReadOnly;
@group(1) @binding(9) var<storage, read> sculpt_instance_data: array<SculptInstanceData>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let instance_data : ptr<storage, ProxyInstanceData, read> = &brick_buffers.preview_instance_data[in.instance_id];

    var vertex_in_sculpt_space : vec3f = in.position * BRICK_WORLD_SIZE * 0.5 + instance_data.position;
    var vertex_in_world_space : vec4f = (sculpt_instance_data[preview_stroke.current_sculpt_idx].model * vec4f(vertex_in_sculpt_space, 1.0));

    // let model_mat = mat4x4f(vec4f(BOX_SIZE, 0.0, 0.0, 0.0), vec4f(0.0, BOX_SIZE, 0.0, 0.0), vec4f(0.0, 0.0, BOX_SIZE, 0.0), vec4f(instance_pos.x, instance_pos.y, instance_pos.z, 1.0));

    var out: VertexOutput;
    out.position = camera_data.view_projection * vertex_in_world_space;
    out.is_interior = instance_data.in_use;
    //out.edit_range = vec2u(preview_stroke.stroke.edit_list_index, preview_stroke.stroke.edit_count);
    out.edit_range = vec2u(instance_data.edit_id_start, instance_data.edit_count);
    
    out.vertex_in_sculpt_space = vertex_in_sculpt_space;
    // This is in an attribute for debugging
    out.vertex_in_world_space = vertex_in_world_space.xyz;
    out.voxel_center_sculpt_space = instance_data.position;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32
}

#define MAX_LIGHTS

@group(2) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(2) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(2) @binding(2) var sampler_clamp: sampler;
@group(2) @binding(3) var<uniform> lights : array<Light, MAX_LIGHTS>;
@group(2) @binding(4) var<uniform> num_lights : u32;

fn get_material_preview() -> SdfMaterial {
    var material : SdfMaterial;
    material.albedo = preview_stroke.stroke.material.color.xyz;
    material.roughness = preview_stroke.stroke.material.roughness;
    material.metallic = preview_stroke.stroke.material.metallic;
    return material;
}

fn sample_sdf_preview(position : vec3f) -> f32
{
    // TODO: preview edits
    var material : SdfMaterial = get_material_preview();
    var surface : Surface;
    if (is_inside_brick) {
        surface.distance = -10000.0;
    } else {
        surface.distance = 10000.0;
    }
    
    surface = evaluate_stroke(position, &(preview_stroke.stroke), &(preview_stroke.edit_list), surface, edit_range.x, edit_range.y);
    
    return surface.distance;
}

var<private> last_found_surface_distance : f32;

// https://iquilezles.org/articles/normalsSDF/
fn estimate_normal_preview(sculpt_position : vec3f) -> vec3f
{
    let k : vec2f = vec2f(1.0, -1.0);
    return normalize( k.xyy * sample_sdf_preview( sculpt_position + k.xyy * DERIVATIVE_STEP) + 
                      k.yyx * sample_sdf_preview( sculpt_position + k.yyx * DERIVATIVE_STEP) + 
                      k.yxy * sample_sdf_preview( sculpt_position + k.yxy * DERIVATIVE_STEP) + 
                      k.xxx * sample_sdf_preview( sculpt_position + k.xxx * DERIVATIVE_STEP) );
}

fn raymarch_sculpt_space(ray_origin_sculpt_space : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
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
		pos = ray_origin_sculpt_space + ray_dir * depth;

        distance = sample_sdf_preview(pos);

		if (distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 

        depth += distance;
	}

    if (exit == 1u) {

        let model_matrix = sculpt_instance_data[preview_stroke.current_sculpt_idx].model;
        let position_in_world : vec3f = (model_matrix * vec4f(pos, 1.0)).xyz;

        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(position_in_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal_preview(pos);
        let normal_world : vec3f = normalize(adjoint(model_matrix) * normal);
        let ray_dir_world : vec3f = normalize(adjoint(model_matrix) * ray_dir);

        let material : SdfMaterial = get_material_preview();
        //let material : SdfMaterial = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
        return vec4f(apply_light(-ray_dir_world, pos, position_in_world, normal_world, lightPos + lightOffset, material), depth);
        //return vec4f(vec3f(material.albedo), depth);
        //return vec4f(normal *0.5 + 0.50, depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(vec3f(0.0), 0.999);
}

var<private> is_inside_brick : bool;
var<private> edit_range : vec2u;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;

    let camera_to_vertex = in.vertex_in_world_space.xyz - camera_data.eye;
    let camera_to_vertex_distance = length(camera_to_vertex);

    edit_range = in.edit_range;

    let ray_dir_world : vec3f = camera_to_vertex / camera_to_vertex_distance;
    let ray_dir_sculpt : vec3f = (sculpt_instance_data[preview_stroke.current_sculpt_idx].inv_model * vec4f(ray_dir_world, 0.0)).xyz;

    let raymarch_distance : f32 = min(
        camera_to_vertex_distance,
        ray_intersect_AABB_only_near(in.vertex_in_sculpt_space.xyz, -ray_dir_sculpt, in.voxel_center_sculpt_space, vec3f(BRICK_WORLD_SIZE))
    );

    let ray_origin : vec3f = in.vertex_in_sculpt_space.xyz + raymarch_distance * (-ray_dir_sculpt);

    is_inside_brick = (in.is_interior & INTERIOR_BRICK_FLAG) == INTERIOR_BRICK_FLAG;

    let ray_result = raymarch_sculpt_space(ray_origin, ray_dir_sculpt, raymarch_distance, camera_data.view_projection);

    var final_color : vec3f = ray_result.rgb; 

    if (GAMMA_CORRECTION == 1) {
        final_color = pow(final_color, vec3(1.0 / 2.2));
    }

    out.color = vec4f(final_color, 1.0); // Color
    out.depth = ray_result.a;

    let interpolant : f32 = (f32( edit_range.y ) / f32(preview_stroke.stroke.edit_count)) * (M_PI / 2.0);
    var heatmap_color : vec3f;
    heatmap_color.r = sin(interpolant);
    heatmap_color.g = sin(interpolant * 2.0);
    heatmap_color.b = cos(interpolant);

    // if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
    //     if (is_inside_brick) {
    //         out.color = vec4f(1.0, 0.0, 1.0, 1.0);
    //     } else {
    //         out.color = vec4f(heatmap_color.x, heatmap_color.y, heatmap_color.z, 1.0);
    //         //out.color = vec4f(0.0, 0.0, f32(edit_range.y) / f32(preview_stroke.stroke.edit_count), 1.0);
    //     }
        
    //     out.depth = in.position.z;
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