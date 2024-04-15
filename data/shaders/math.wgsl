const M_PI = 3.14159265359;
const M_PI_INV = 1.0 / 3.14159265359;

// fn hamiltonian_product(q1 : vec4f, q2 : vec4f) -> vec4f {
//     var q : vec4f;
// }

fn quat_mult(q1 : vec4f, q2 : vec4f) -> vec4f
{
    return vec4f(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

fn quat_conj(q : vec4f) -> vec4f
{
    return vec4f(-q.x, -q.y, -q.z, q.w);
}

fn quat_inverse(q : vec4f) -> vec4f
{
    let conj : vec4f = quat_conj(q);
    return conj / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

fn rotate_point_quat(position : vec3f, rotation : vec4f) -> vec3f
{
    // q_pos = quat(position)
    // pr_h = hamilton(hamilton(rot, q_pos), conj(rot))
    // return pr_h.xyz
    let q_pos : vec4f = vec4f(position.x, position.y, position.z, 0.0);
    let pr_h = quat_mult(quat_mult(rotation, q_pos), quat_conj(rotation));
    return pr_h.xyz;
    //return position + 2.0 * cross(rotation.xyz, cross(rotation.xyz, position) + rotation.w * position);
}