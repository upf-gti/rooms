struct RayInfo
{
    ray_origin : vec3f,
    dummy0     : f32,
    ray_dir    : vec3f,
    dummy1     : f32,
}

struct RayIntersectionInfo
{
    intersected     : u32,
    tile_pointer    : u32,
    sculpt_id       : u32,
    ray_t           : f32
};

struct RayIntersectionInstances {
    curr_instance_idx : u32,
    instance_indices : array<u32>
};