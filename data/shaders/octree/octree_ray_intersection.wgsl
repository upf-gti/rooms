#include octree_includes.wgsl

struct RayInfo
{
    ray_origin : vec3f,
    dummy0     : f32,
    ray_dir    : vec3f,
    dummy1     : f32
}

@group(0) @binding(2) var<storage, read_write> octree : Octree;
// @group(0) @binding(3) var read_sdf: texture_3d<f32>;
// @group(0) @binding(4) var texture_sampler : sampler;
// @group(0) @binding(5) var<storage, read> octree_proxy_data: OctreeProxyInstancesNonAtomic;

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

    // Check intersection with octree aabb
    if (ray_AABB_intersection(ray_info.ray_origin, ray_info.ray_dir, bounds_min, bounds_max, &t_near, &t_far))
    {
        var parent_octant_id : u32 = 0;

        var level : u32 = 1;
        push_iteration_data(&stack_pointer, level, 0, 0, vec3f(0.0, 0.0, 0.0));

        // Compute the center and the half size of the current octree level
        while (level <= OCTREE_DEPTH && stack_pointer > 0) {

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
                    if (octants_to_visit[j].distance < octants_to_visit[i].distance)
                    {
                        let swap = octants_to_visit[i];
                        octants_to_visit[i] = octants_to_visit[j];
                        octants_to_visit[j] = swap;
                    }
                }
            }

            for (var i : u32 = 0; i < octants_count; i++) {

                let octant_id : u32 = parent_octant_id | (octants_to_visit[i].octant << (3 * (level - 1)));
                let is_last_level : bool = level == OCTREE_DEPTH;

                if (!is_last_level) {
                    push_iteration_data(&stack_pointer, level + 1, 0, octant_id, octants_to_visit[i].octant_center);
                } else {
                    last_level = level;
                    last_octant = octants_to_visit[i].octant;
                    // If the brick is filled
                    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
                    if (octants_to_visit[i].distance < intersected_distance &&
                       (octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
                        intersected = true;
                        ray_intersection_info.tile_pointer = octree.data[octree_index].tile_pointer;
                        intersected_distance = octants_to_visit[i].distance;
                        ray_intersection_info.intersection_center = octants_to_visit[i].octant_center;
                    }
                }
            }
        }

    }

    ray_intersection_info.intersected = select(0u, 1u, intersected);
}
