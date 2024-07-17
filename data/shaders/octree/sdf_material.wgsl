
struct SdfMaterial {
    albedo      : vec3f,
    roughness   : f32,
    metalness   : f32
};

// Material operation storages
fn Material_mult_by(m : SdfMaterial, v : f32) -> SdfMaterial {
    return SdfMaterial(m.albedo * v, m.roughness * v, m.metalness * v);
}

fn Material_sum_Material(m1 : SdfMaterial, m2 : SdfMaterial) -> SdfMaterial {
    return SdfMaterial(m1.albedo + m2.albedo, m1.roughness + m2.roughness, m1.metalness + m2.metalness);
}

fn Material_mix(m1 : SdfMaterial, m2 : SdfMaterial, t : f32) -> SdfMaterial {
    return Material_sum_Material(Material_mult_by(m1, 1.0 - t), Material_mult_by(m2, t));
}
