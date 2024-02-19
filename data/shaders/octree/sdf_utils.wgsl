const DERIVATIVE_STEP = 0.5 / SDF_RESOLUTION;
const MAX_ITERATIONS = 60;

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

fn ray_intersect_AABB_only_near(rayOrigin : vec3f, rayDir: vec3f, box_origin: vec3f, box_size: vec3f) -> f32 {
    let boxMin : vec3f = box_origin - (box_size / 2.0);
    let boxMax : vec3f = boxMin + box_size;

    let tMin : vec3f = (boxMin - rayOrigin) / rayDir;
    let tMax : vec3f = (boxMax - rayOrigin) / rayDir;
    let t2 : vec3f = max(tMin, tMax);
    let tFar : f32 = min(min(t2.x, t2.y), t2.z);
    return tFar;
};