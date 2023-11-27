#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl

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
    out.color = in.color;
    out.normal = in.normal;
    
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

const DERIVATIVE_STEP = 0.5 / SDF_RESOLUTION;
const MAX_ITERATIONS = 60;

const specularCoeff = 1.0;
const specularExponent = 4.0;
const lightPos = vec3f(0.0, 2.0, 1.0);

// SKYMAP FUNCTION
fn irradiance_spherical_harmonics(n : vec3f) -> vec3f {
    return vec3f(0.366, 0.363, 0.371)
        + vec3f(0.257, 0.252, 0.263) * (n.y)
        + vec3f(0.108, 0.109, 0.113) * (n.z)
        + vec3f(0.028, 0.044, 0.05) * (n.x)
        + vec3f(0.027, 0.038, 0.036) * (n.y * n.x)
        + vec3f(0.11, 0.11, 0.118) * (n.y * n.z)
        + vec3f(-0.11, -0.113, -0.13) * (3.0 * n.z * n.z - 1.0)
        + vec3f(0.016, 0.018, 0.016) * (n.z * n.x)
        + vec3f(-0.033, -0.033, -0.037) * (n.x * n.x - n.y * n.y);
}

// https://community.khronos.org/t/ray-vs-aabb-exit-point-knowing-entry-point/66307/3
fn ray_AABB_intersection_distance(ray_origin : vec3f,
                                  ray_dir : vec3f,
                                  box_origin : vec3f,
                                  box_size : vec3f) -> f32 {
    let box_min : vec3f = box_origin - (box_size / 2.0);
    let box_max : vec3f = box_min + box_size;

    let min_max : array<vec3f, 2> = array<vec3f, 2>(box_min, box_max);

    var tmax : vec3f;
	let div : vec3f = 1.0 / ray_dir;
	let indexes : vec3i = vec3i(i32(step(0.0, div.x)), i32((step(0.0, div.y))), i32(step(0.0, div.z)));
	tmax.x = (min_max[indexes[0]].x - ray_origin.x) * div.x;
	tmax.y = (min_max[indexes[1]].y - ray_origin.y) * div.y;
	tmax.z = (min_max[indexes[2]].z - ray_origin.z) * div.z;

	return min(min(tmax.x, tmax.y), tmax.z);
}

fn sample_color(pos : vec3u) -> Material {
    let sample : u32 = textureLoad(read_material_sdf, pos, 0).r;

    return unpack_material(sample);
}

// From: http://paulbourke.net/miscellaneous/interpolation/
fn interpolate_material(pos : vec3f) -> Material {
    var result : Material;

    let pos_f_part : vec3f = abs(fract(pos));
    let pos_i_part : vec3u = vec3u(floor(pos)) - vec3u(1);

    let index000 : vec3u = pos_i_part;
    let index010 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 1, pos_i_part.z + 0)  ;
    let index100 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 0, pos_i_part.z + 0)  ;
    let index001 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 0, pos_i_part.z + 1)  ;
    let index101 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 0, pos_i_part.z + 1)  ;
    let index011 : vec3u = vec3u(pos_i_part.x + 0, pos_i_part.y + 1, pos_i_part.z + 1)  ;
    let index110 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 1, pos_i_part.z + 0)  ;
    let index111 : vec3u = vec3u(pos_i_part.x + 1, pos_i_part.y + 1, pos_i_part.z + 1)  ;

    result = Material_mult_by(sample_color(index000), (1.0 - pos_f_part.x) * (1.0 - pos_f_part.y) * (1.0 - pos_f_part.z));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index100), (pos_f_part.x) * (1.0 - pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index010), (1.0 - pos_f_part.x) * (pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index001), (1.0 - pos_f_part.x) * (1.0 - pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index101), (pos_f_part.x) * (1.0 - pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index011), (1.0 - pos_f_part.x) * (pos_f_part.y) * (pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index110), (pos_f_part.x) * (pos_f_part.y) * (1.0 - pos_f_part.z)));
    result = Material_sum_Material(result, Material_mult_by(sample_color(index111), (pos_f_part.x) * (pos_f_part.y) * (pos_f_part.z)));

    return result;
}

fn sample_sdf(position : vec3f) -> Surface
{
    let rot_p = rotate_point_quat(position - sculpt_data.sculpt_start_position, sculpt_data.sculpt_inv_rotation);

    // if (rot_p.x < 0.0 || rot_p.x > 1.0 ||
    //     rot_p.y < 0.0 || rot_p.y > 1.0 ||
    //     rot_p.z < 0.0 || rot_p.z > 1.0) {
    //     return Surface(vec3(0.0, 0.0, 0.0), 0.01);
    // }

    let data : vec4f = textureSampleLevel(read_sdf, texture_sampler, position, 0.0);

    var surface : Surface = Surface(vec3f(1.0, 0.0, 0.0), data.r);

    //surface = add_preview_edits((p  + compute_data.sculpt_start_position) * SDF_RESOLUTION, surface);

    return surface;
}

// https://iquilezles.org/articles/normalsSDF/
fn estimate_normal( p : vec3f) -> vec3f
{
    let k : vec2f = vec2f(1.0, -1.0);
    return normalize( k.xyy * sample_sdf( p + k.xyy * DERIVATIVE_STEP ).distance + 
                      k.yyx * sample_sdf( p + k.yyx * DERIVATIVE_STEP ).distance + 
                      k.yxy * sample_sdf( p + k.yxy * DERIVATIVE_STEP ).distance + 
                      k.xxx * sample_sdf( p + k.xxx * DERIVATIVE_STEP ).distance );
}

// TODO: if diffuse variable is not used, performance is increased by 20% (????)
fn blinn_phong(toEye : vec3f, position : vec3f, position_world : vec3f, lightPosition : vec3f, ambient : vec3f, diffuse : vec3f) -> vec3f
{
    var normal : vec3f = estimate_normal(position);
    normal = normalize(rotate_point_quat(normal, sculpt_data.sculpt_rotation));

    let toLight : vec3f = normalize(lightPosition - position_world);
    let reflection : vec3f = normalize(reflect(-toLight, normal));
    let halfwayDir : vec3f = normalize(toLight + toEye);

    let ambientFactor : vec3f = ambient * diffuse;
    let diffuseFactor : vec3f = 0.4 * diffuse * max(0.0, dot(normal, toLight));
    let specularFactor : vec3f = vec3f(0.3) * pow(max(0.0, dot(toEye, reflection)), specularExponent);

    return ambientFactor + diffuseFactor + specularFactor;
    //return normal;
    //return diffuse;
}


fn raymarch(ray_origin : vec3f, ray_origin_world : vec3f, ray_dir : vec3f, max_distance : f32, view_proj : mat4x4f) -> vec4f
{
    let ambientColor = vec3f(0.4);
	let hitColor = vec3f(1.0, 1.0, 1.0);
	let missColor = vec3f(0.0, 0.0, 0.0);
    let lightOffset = vec3f(0.0, 0.0, 0.0);

	var depth : f32 = 0.0;
    var surface : Surface;

    var pos : vec3f;
    var i : i32 = 0;
    var exit : u32 = 0u;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		pos = ray_origin + ray_dir * depth;

        surface = sample_sdf(pos);

		if (surface.distance < MIN_HIT_DIST) {
            exit = 1u;
            break;
		} 

        depth += (surface.distance) * SCALE_CONVERSION_FACTOR;
	}

    if (exit == 1u) {
        let pos_world : vec3f = ray_origin_world + ray_dir * (depth / SCALE_CONVERSION_FACTOR);
        let epsilon : f32 = 0.000001; // avoids flashing when camera inside sdf
        let proj_pos : vec4f = view_proj * vec4f(pos_world + ray_dir * epsilon, 1.0);
        depth = proj_pos.z / proj_pos.w;

        let normal : vec3f = estimate_normal(pos);

        let material : Material = interpolate_material(pos * SDF_RESOLUTION);
        //let material : Material = interpolate_material((pos - normal * 0.001) * SDF_RESOLUTION);
        //return vec4f(surface.color, depth);
		return vec4f(blinn_phong(-ray_dir, pos, pos_world, lightPos + lightOffset, ambientColor, material.albedo), depth);
	}

    // Use a two band spherical harmonic as a skymap
    return vec4f(irradiance_spherical_harmonics(ray_dir.xzy), 0.999);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    var out: FragmentOutput;
    let ray_dir : vec3f = normalize(in.world_pos.xyz - eye_position);
    let ray_dir_voxel_space : vec3f = normalize(in.voxel_pos - rotate_point_quat(eye_position - sculpt_data.sculpt_start_position, sculpt_data.sculpt_inv_rotation));

    let raymarch_distance : f32 = ray_AABB_intersection_distance(in.voxel_pos, ray_dir_voxel_space, in.voxel_center, vec3f(BRICK_WORLD_SIZE));

    let ray_result = raymarch(in.in_atlas_pos.xyz, in.world_pos.xyz, ray_dir_voxel_space, raymarch_distance * SCALE_CONVERSION_FACTOR, camera_data.view_projection);

    out.color = vec4f(pow(ray_result.rgb, vec3f(2.2, 2.2, 2.2)), 1.0); // Color
    out.depth = ray_result.a;

    // if ( in.uv.x < 0.015 || in.uv.y > 0.985 || in.uv.x > 0.985 || in.uv.y < 0.015 )  {
    //     out.color = vec4f(0.0, 0.0, 0.0, 1.0);
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