var<private> thread_stroke_index_buffer : array<u32, (MAX_STROKE_INFLUENCE_COUNT >> 1u) >;
var<private> thread_stroke_index_count : u32 = 0u;

// OJO: condiciones de carrera. Para evitarlo, los alloc en este buffer se hacen con padding de u32
// NOTE: i >> 1u == i / 2u

fn StrokeCulling_get_stroke_index_at(buffer_index : u32, curr_level : u32) -> u32 {
    var buffer_value : u32 = stroke_culling.stroke_index_buffer[(buffer_index >> 1u) + ((curr_level % 2u) * STROKE_INDEX_HALF_SIZE)];

    // Test if its even or uneven to choose the first 16 bits of teh rest of the u32
    if ((buffer_index & 0x1u) == 0x1u) {
        buffer_value = buffer_value & 0xFFFFu;
    } else {
        buffer_value = buffer_value >> 16u;
    }

    return buffer_value;
}

fn StrokeCulling_set_stroke_index_at(stroke_index : u32, buffer_index : u32, curr_level : u32) {
    let real_buffer_index : u32 = (buffer_index >> 1u) + ((curr_level % 2u) * STROKE_INDEX_HALF_SIZE);
    var buffer_data : u32 = stroke_culling.stroke_index_buffer[real_buffer_index];

    if ((buffer_index & 0x1u) == 0x1u) {
        buffer_data = buffer_data | (0xFFFFu & stroke_index);
    } else  {
        buffer_data = buffer_data | (stroke_index << 16u);
    }

    stroke_culling.stroke_index_buffer[real_buffer_index] = buffer_data;
}

fn StrokeCulling_alloc_stroke_index_buffer(index_to_alloc_count : u32) -> u32 {
    return atomicAdd(&stroke_culling.stroke_index_count, index_to_alloc_count >> 1u);
}

// Bounds (Base index, index count)
fn StrokeCulling_get_index_bounds_of_node(node_id : u32, curr_level : u32) -> vec2u {
    // Each thread writes to each of segment, so in order to avoid race conditions
    // the memory reserve size is Nxu32
    let raw_indices_data : u32 = stroke_culling.stroke_indices_per_node_buffer[node_id + ((curr_level % 2u) * MAX_SUBDIVISION_SIZE)];

    return vec2u(raw_indices_data >> 16u, raw_indices_data & 0xFFFFu);
}

fn StrokeCulling_set_index_bounds_of_node(node_id : u32, stroke_buffer_idx : u32, stroke_buffer_size : u32, curr_level : u32) {
    let to_store_data : u32 = (stroke_buffer_idx << 16u) | (stroke_buffer_size & 0xFFFFu);

    stroke_culling.stroke_indices_per_node_buffer[node_id + ((curr_level % 2u) * MAX_SUBDIVISION_SIZE)] = to_store_data;
}

fn add_index_to_thread_index_buffer(index : u32) {
    let real_buffer_index : u32 = (thread_stroke_index_count >> 1u);
    var buffer_data : u32 = 0u;

    if ((thread_stroke_index_count & 0x1u) == 0x1u) {
        buffer_data = thread_stroke_index_buffer[real_buffer_index];
        buffer_data = buffer_data | (0xFFFFu & index);
    } else  {
        buffer_data = (index << 16u);
    }

    thread_stroke_index_buffer[real_buffer_index] = buffer_data;
    thread_stroke_index_count++;
}

fn StrokeCulling_copy_thread_stroke_buffer_to_common_buffer(node_id : u32, level : u32) {
    // Write to the shared index buffer the indices of the strokes in this thread
    let starting_idx : u32 = StrokeCulling_alloc_stroke_index_buffer(thread_stroke_index_count);
    let packed_size : u32 = thread_stroke_index_count >> 1u;
    let margin : u32 = ((level % 2u) * STROKE_INDEX_HALF_SIZE);

    // Write the packed u16 indices
    for(var i : u32 = 0u; i <= packed_size; i++) {
        stroke_culling.stroke_index_buffer[starting_idx + i + margin] = thread_stroke_index_buffer[i];
    }

    // Store the (starding, size) tuple for searching on the index buffer
    StrokeCulling_set_index_bounds_of_node(node_id, starting_idx, thread_stroke_index_count, level);
}