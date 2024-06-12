#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> brick_copy_buffer : array<u32>;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers;

@group(1) @binding(0) var<storage, read> frustrum_data : FrustrumCullingData;

/**
    Este shader itera por todo el buffer de render instances de bricks, y en funcion
    de si esta lleno o vacio, incrementa el numero de instancias del indirect buffer
    para el instancing de los bricks.
    Tambien se rellena un buffer que relaciona cada instancia con un indice del instancing.
*/

// https://bartwronski.com/2017/04/13/cull-that-cone/
fn sphere_cone_culling(sphere_center : vec3f, sphere_radius : f32) -> bool {
    let sphere_cone_distance : f32 = sphere_radius + frustrum_data.cone_distance;

    // Cone 1
    let v1 : vec3f = sphere_center - frustrum_data.cone1_origen;
    let v1_len_sq : f32 = dot(v1, v1);
    let v1_len1 : f32 = dot(v1, frustrum_data.cone1_normal);
    let dist_closest_point1 : f32 = frustrum_data.cos_angle * sqrt(v1_len_sq - v1_len1 * v1_len1) - v1_len1 * frustrum_data.sin_angle;

    let angle_cull1 : bool = dist_closest_point1 > sphere_radius;
    let front_cull1 : bool = v1_len1 > (sphere_cone_distance);
    let back_cull1 : bool = v1_len1 < -sphere_radius;

    // Cone 2
    let v2 : vec3f = sphere_center - frustrum_data.cone2_origen;
    let v2_len_sq : f32 = dot(v2, v2);
    let v2_len1 : f32 = dot(v2, frustrum_data.cone2_normal);
    let dist_closest_point2 : f32 = frustrum_data.cos_angle * sqrt(v2_len_sq - v2_len1 * v2_len1) - v2_len1 * frustrum_data.sin_angle;

    let angle_cull2 : bool = dist_closest_point2 > sphere_radius;
    let front_cull2 : bool = v2_len1 > (sphere_cone_distance);
    let back_cull2 : bool = v2_len1 < -sphere_radius;

    return !(angle_cull1 || front_cull1 || back_cull1) || !(angle_cull2 || front_cull2 || back_cull2);
}

@compute @workgroup_size(8u, 8u, 8u)
fn compute(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) local_id: u32)
{
    // 512 is 8 x 8 x 8, which is the number of threads in a group
    let current_instance_index : u32 = (id.x) * 512u + local_id;

    let current_instance_in_use_flag : u32 = brick_buffers.brick_instance_data[current_instance_index].in_use;

    if ((current_instance_in_use_flag & BRICK_IN_USE_FLAG) == BRICK_IN_USE_FLAG
     && (current_instance_in_use_flag & BRICK_HIDE_FLAG) == 0
     && sphere_cone_culling(brick_buffers.brick_instance_data[current_instance_index].position, BRICK_WORLD_SIZE / 2.0))
    {
        let prev_value : u32 = atomicAdd(&brick_buffers.brick_instance_counter, 1u);
        brick_copy_buffer[prev_value] = current_instance_index;
    }
}
