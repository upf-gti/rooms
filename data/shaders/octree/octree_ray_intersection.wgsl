#include octree_includes.wgsl

struct RayInfo
{
    ray_origin : vec3f,
    dummy0     : f32,
    ray_dir    : vec3f,
    dummy1     : f32
}

struct RayIntersectionInfo
{
    intersected : u32
}

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var read_sdf: texture_3d<f32>;
@group(0) @binding(4) var texture_sampler : sampler;
@group(0) @binding(5) var<storage, read> octree_proxy_data: OctreeProxyInstancesNonAtomic;

@group(1) @binding(0) var<uniform> ray_info: RayInfo;
@group(1) @binding(1) var<storage, read_write> ray_intersection_info: RayIntersectionInfo;

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

@compute @workgroup_size(1, 1, 1)
fn compute()
{
    let bounds_min : vec3f = vec3f(-SCULPT_MAX_SIZE) * 0.5;
    let bounds_max : vec3f = vec3f(SCULPT_MAX_SIZE) * 0.5;
    let cells_per_side : vec4f = vec4f(trunc((bounds_max - bounds_min) / BRICK_WORLD_SIZE), 1);

    var t_near : f32;
    var t_far : f32;

    let node : OctreeNode = octree.data[0];
    let sdf : f32 = textureSampleLevel(read_sdf, texture_sampler, vec3f(0.0, 0.0, 0.0), 0.0).r;
    let instance_data : ProxyInstanceData = octree_proxy_data.instance_data[0];

    var intersected : bool = false;

    // Check intersection with octree aabb
    if (ray_AABB_intersection(ray_info.ray_origin, ray_info.ray_dir, bounds_min, bounds_max, &t_near, &t_far))
    {

        var level : u32 = 1;

        var current_ray_origin : vec3f = ray_info.ray_origin + ray_info.ray_dir * t_near;

        var octant_id = 0; //octant_id | (i << (3 * level));

        // Compute the center and the half size of the current octree level
        for (var level : u32 = 1; level <= OCTREE_DEPTH;) {

            var octant_center : vec3f = vec3f(0.0);

            let level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(level + 1));

            for (var child : u32 = 0; child < 8; child++) {
                octant_center += level_half_size * OCTREE_CHILD_OFFSET_LUT[child];

                if (ray_AABB_intersection(current_ray_origin, ray_info.ray_dir, octant_center - level_half_size, octant_center + level_half_size, &t_near, &t_far))
                {
                    if (level < OCTREE_DEPTH) {
                        level++;
                        break;
                    } else {
                        intersected = true;
                    }
                }

                if (child == 7) {
                    // level--;
                }
            }
        }

    }

    ray_intersection_info.intersected = select(0u, 1u, intersected);
}