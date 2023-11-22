

struct Material {
    albedo : vec3f,
    roughness : f32,
    metalness : f32
};

/*
    RGB: R 7 bits + B 8 bits + G 7 bits = 22 bits
    Metalness: 5 bits
    Roughness: 5 bits
*/

const RED_MASK : u32 = 0x7f << 25u;//0xFE000000u;
const GREEN_MASK : u32 = 0xFF << 17u; // 0x1FE0000u;
const BLUE_MASK : u32 = 0x7f << 10u; // 0x1FC00u;


fn unpack_material(packed_material : u32) -> Material {
    var resulting_material : Material;

    let red_val : u32 = (packed_material & RED_MASK) >> 25u;
    let green_val : u32 = (packed_material & GREEN_MASK) >> 17u;
    let blue_val : u32 = (packed_material & BLUE_MASK) >> 10u;

    resulting_material.albedo = vec3f(f32(red_val) / 127.0, f32(green_val) / 255.0, f32(blue_val) / 127.0);

    return resulting_material;
}

fn pack_material(material : Material) -> u32 {
    var packed_material : u32 = 0u;

    let red : u32 = u32(material.albedo.r * 127.0);
    let green : u32 = u32(material.albedo.g * 255.0);
    let blue : u32 = u32(material.albedo.b * 127.0);

    packed_material = red << 25u;
    packed_material |= green << 17u;
    packed_material |= blue << 10u;

    return packed_material;
}