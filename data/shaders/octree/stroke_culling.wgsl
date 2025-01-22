fn culling_get_culling_data(stroke_pointer : u32, edit_start : u32, edit_count : u32) -> u32 {
    var result : u32 = stroke_pointer << 16;
    // result |= (edit_start & 0xFFu) << 8;
    // result |= (edit_count & 0xFFu);

    return result;
}

fn culling_get_stroke_index(culling_data : u32) -> u32 {
    return (culling_data) >> 16;
}