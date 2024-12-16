const DERIVATIVE_STEP = 0.5 / SDF_RESOLUTION;
const MAX_ITERATIONS = 60;
const MIN_HIT_DIST = 0.0005;
const MAX_HIT_DIST = 0.005;


fn ray_intersect_AABB_only_near(rayOrigin : vec3f, rayDir: vec3f, box_origin: vec3f, box_size: vec3f) -> f32 {
    let box_min : vec3f = box_origin - (box_size / 2.0);
    let box_max : vec3f = box_min + box_size;

    let tMin : vec3f = (box_min - rayOrigin) / rayDir;
    let tMax : vec3f = (box_max - rayOrigin) / rayDir;
    let t2 : vec3f = max(tMin, tMax);
    let tFar : f32 = min(min(t2.x, t2.y), t2.z);
    return tFar;
};