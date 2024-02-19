#include octree_includes.wgsl

struct RayInfo
{
    ray_origin : vec3f,
    dummy0     : f32,
    ray_dir    : vec3f,
    dummy1     : f32
}

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(4) var texture_sampler : sampler;
@group(0) @binding(8) var read_material_sdf: texture_3d<u32>;

@group(1) @binding(0) var<uniform> ray_info: RayInfo;
@group(1) @binding(3) var<storage, read_write> ray_intersection_info: RayIntersectionInfo;

// https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
fn ray_AABB_intersection(ray_origin : vec3f, ray_dir : vec3f, box_min : vec3f, box_max : vec3f, t_near : ptr<function, f32>, t_far : ptr<function, f32>) -> bool
{
    let t_min : vec3f = (box_min - ray_origin) / ray_dir;
    let t_max : vec3f = (box_max - ray_origin) / ray_dir;

    let t1 : vec3f = min(t_min, t_max);
    let t2 : vec3f = max(t_min, t_max);

    *t_near = max(max(t1.x, t1.y), t1.z);
    *t_far = min(min(t2.x, t2.y), t2.z);

    return *t_near <= *t_far && *t_far >= 0;
};

struct IterationData {
    level : u32,
    octant : u32,
    octant_id : u32,
    octant_center : vec3f
}

struct VisitedOctantData {
    octant: u32,
    distance : f32,
    octant_center : vec3f
}

var<workgroup> iteration_data_stack: array<IterationData, 100>;

fn push_iteration_data(stack_pointer : ptr<function, u32>, level : u32, octant : u32, octant_id : u32, octant_center : vec3f)
{
    iteration_data_stack[*stack_pointer].level = level;
    iteration_data_stack[*stack_pointer].octant = octant;
    iteration_data_stack[*stack_pointer].octant_id = octant_id;
    iteration_data_stack[*stack_pointer].octant_center = octant_center;
    *stack_pointer++;
}

fn pop_iteration_data(stack_pointer : ptr<function, u32>) -> IterationData
{
    *stack_pointer--;
    return iteration_data_stack[*stack_pointer];
}



fn sample_sdf_atlas(atlas_position : vec3f) -> f32
{
    return textureSampleLevel(read_sdf, texture_sampler, atlas_position, 0.0).r / SCULPT_MAX_SIZE;
}

#include sdf_utils.wgsl

// TODO: LOTS of duplication ahead, because current inludes will bloat this file. Need restructuring!
struct Material {
    albedo      : vec3f,
    roughness   : f32,
    metalness   : f32
};

#include material_packing.wgsl

fn sample_material_raw(pos : vec3u) -> Material {
    let sample : u32 = textureLoad(read_material_sdf, pos, 0).r;

    return unpack_material(sample);
}

// Material operation functions
fn Material_mult_by(m : Material, v : f32) -> Material {
    return Material(m.albedo * v, m.roughness * v, m.metalness * v);
}

fn Material_sum_Material(m1 : Material, m2 : Material) -> Material {
    return Material(m1.albedo + m2.albedo, m1.roughness + m2.roughness, m1.metalness + m2.metalness);
}

fn Material_mix(m1 : Material, m2 : Material, t : f32) -> Material {
    return Material_sum_Material(Material_mult_by(m1, 1.0 - t), Material_mult_by(m2, t));
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

fn raymarch(ray_origin_in_atlas_space : vec3f, ray_dir : vec3f, max_distance : f32, has_hit: ptr<function, bool>) -> f32
{
	var depth : f32 = 0.0;
    var distance : f32;

    var position_in_atlas : vec3f;
    var i : i32 = 0;

    *has_hit = false;

	for (i = 0; depth < max_distance && i < MAX_ITERATIONS; i++)
    {
		position_in_atlas = ray_origin_in_atlas_space + ray_dir * depth;

        distance = sample_sdf_atlas(position_in_atlas);

		if (distance < MIN_HIT_DIST) {
            *has_hit = true;
            return depth;
		} 

        depth += distance;
	}

    return -1000.0;
}


@compute @workgroup_size(1, 1, 1)
fn compute()
{
    let bounds_min : vec3f = vec3f(-SCULPT_MAX_SIZE) * 0.5;
    let bounds_max : vec3f = vec3f(SCULPT_MAX_SIZE) * 0.5;
    let cells_per_side : vec4f = vec4f(trunc((bounds_max - bounds_min) / BRICK_WORLD_SIZE), 1);

    var t_near : f32;
    var t_far : f32;

    var intersected : bool = false;

    var stack_pointer : u32 = 0;

    var last_level : u32 = 0;
    var last_octant : u32 = 0;

    var intersected_distance : f32 = 10000000.0;
    ray_intersection_info.intersected = 0u;
    ray_intersection_info.tile_pointer = 0u;

    // Check intersection with octree aabb
    if (ray_AABB_intersection(ray_info.ray_origin, ray_info.ray_dir, bounds_min, bounds_max, &t_near, &t_far))
    {
        var parent_octant_id : u32 = 0;

        var level : u32 = 1;
        push_iteration_data(&stack_pointer, level, 0, 0, vec3f(0.0, 0.0, 0.0));

        // Compute the center and the half size of the current octree level
        while (level <= OCTREE_DEPTH && stack_pointer > 0 && !intersected) {

            let iteration_data : IterationData = pop_iteration_data(&stack_pointer);

            level = iteration_data.level;
            parent_octant_id = iteration_data.octant_id;
            let parent_octant_center : vec3f = iteration_data.octant_center;

            let level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(level + 1));

            var octants_to_visit : array<VisitedOctantData, 8>;
            var octants_count : u32 = 0;

            for (var octant : u32 = 0; octant < 8; octant++) {
                let octant_center = parent_octant_center + level_half_size * OCTREE_CHILD_OFFSET_LUT[octant];

                if (ray_AABB_intersection(ray_info.ray_origin, ray_info.ray_dir, octant_center - level_half_size, octant_center + level_half_size, &t_near, &t_far))
                {
                    octants_to_visit[octants_count].octant = octant;
                    octants_to_visit[octants_count].distance = t_near;
                    octants_to_visit[octants_count].octant_center = octant_center;
                    octants_count++;
                }
            }

            // sort by distance
            for (var i : u32 = 0; i < octants_count; i++) {
                for(var j = i; j < octants_count; j++) {
                    if (octants_to_visit[j].distance > octants_to_visit[i].distance)
                    {
                        let swap = octants_to_visit[i];
                        octants_to_visit[i] = octants_to_visit[j];
                        octants_to_visit[j] = swap;
                    }
                }
            }

            for (var i : u32 = 0; i < octants_count && !intersected; i++) {

                let octant_id : u32 = parent_octant_id | (octants_to_visit[i].octant << (3 * (level - 1)));
                let is_last_level : bool = level == OCTREE_DEPTH;

                if (!is_last_level) {
                    push_iteration_data(&stack_pointer, level + 1, 0, octant_id, octants_to_visit[i].octant_center);
                } else {
                    last_level = level;
                    last_octant = octants_to_visit[i].octant;
                    // If the brick is filled
                    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
                    if ((octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
                        intersected_distance = octants_to_visit[i].distance;

                        let atlas_tile_index : u32 = octree.data[octree_index].tile_pointer & OCTREE_TILE_INDEX_MASK;
                        let in_atlas_tile_coordinate : vec3f = vec3f(10 * vec3u(atlas_tile_index % BRICK_COUNT,
                                                  (atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   atlas_tile_index / (BRICK_COUNT * BRICK_COUNT))) / SDF_RESOLUTION;

                        // Ray intersection in sculpt space
                        let in_sculpture_point : vec3f = ray_info.ray_origin + ray_info.ray_dir * octants_to_visit[i].distance;
                        var in_atlas_position : vec3f = in_sculpture_point;
                        // Sculpt coords 2 brick data
                        in_atlas_position -= octants_to_visit[i].octant_center;
                        in_atlas_position *= SCULPT_TO_ATLAS_CONVERSION_FACTOR;
                        in_atlas_position += in_atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION);
                        // From sculpt to  atlas space: (sculpt - brick_center) * SCULPT_TO_ATLAS + atlas_origin

                        let raymarch_max_distance : f32 = ray_intersect_AABB_only_near(in_atlas_position, ray_info.ray_dir, in_atlas_tile_coordinate + vec3f(5.0 / SDF_RESOLUTION), vec3f(BRICK_ATLAS_SIZE));

                        // Raymarching
                        let raymarch_result_distance = raymarch(in_atlas_position, ray_info.ray_dir, raymarch_max_distance, &intersected);

                        if (intersected) {
                            let atlas_position : vec3f = in_atlas_position + ray_info.ray_dir * raymarch_result_distance;
                            let material : Material = sample_material_atlas(atlas_position);
                            ray_intersection_info.tile_pointer = octree.data[octree_index].tile_pointer;

                            ray_intersection_info.material_albedo = material.albedo;   
                            ray_intersection_info.material_roughness = material.roughness;   
                            ray_intersection_info.material_metalness = material.metalness;                           
                            ray_intersection_info.intersected = 1u;
                            ray_intersection_info.intersection_position = in_sculpture_point + ray_info.ray_dir * (raymarch_result_distance / SCULPT_TO_ATLAS_CONVERSION_FACTOR);//octants_to_visit[i].octant_center;
                            return;
                        }
                    }
                }
            }
        }

    }

    
}
