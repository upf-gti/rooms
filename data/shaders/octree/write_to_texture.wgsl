#include sdf_functions.wgsl
#include octree_includes.wgsl

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> current_level : atomic<u32>;
@group(0) @binding(5) var<storage, read_write> atomic_counter : atomic<u32>;
@group(0) @binding(6) var<storage, read_write> proxy_box_position_buffer: array<ProxyInstanceData>;
@group(0) @binding(7) var<storage, read_write> edit_culling_lists: array<u32>;
@group(0) @binding(8) var<storage, read_write> atlas_tile_counter : atomic<u32>;
@group(0) @binding(9) var<storage, read_write> edit_culling_count : array<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

@compute @workgroup_size(10,10,10)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;

    let atlas_tile_index : u32 = proxy_box_position_buffer[atomicLoad(&atlas_tile_counter) - atomicLoad(&atomic_counter) + id].atlas_tile_index;

    let atlas_tile_coordinate : vec3u = 10 * vec3u(atlas_tile_index % BRICK_COUNT,
                                                  (atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   atlas_tile_index / (BRICK_COUNT * BRICK_COUNT));

    let level : u32 = atomicLoad(&current_level);
    
    let parent_level : u32 = level - 1;

    let octant_id : u32 = octant_usage_read[id];
    let parent_mask : u32 = u32(pow(2, f32(merge_data.max_octree_depth * 3))) - 1;
    let parent_octant_id : u32 = octant_id &  (parent_mask >> (3u * (merge_data.max_octree_depth - parent_level)));

    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
    let parent_octree_index : u32 = parent_octant_id + u32((pow(8.0, f32(parent_level)) - 1) / 7);

    var octant_center : vec3f = vec3f(0.0);

    var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));

        octant_center += level_half_size * OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x7];
    }

    // To start writing at the top corner
    let octant_corner : vec3f = octant_center - vec3f(level_half_size);

    var sSurface : Surface = Surface(vec3f(0.0, 0.0, 0.0), 10000.0);

    let pixel_offset : vec3f = (vec3f(local_id) - 1.0) / SDF_RESOLUTION;

    var current_edit_surface : Surface;

    let packed_list_size : u32 = (256 / 4);

    for (var i : u32 = 0; i < edit_culling_count[parent_octree_index]; i++) {

        let current_packed_edit_idx : u32 = edit_culling_lists[i / 4 + parent_octree_index * packed_list_size];

        let packed_index : u32 = 3 - (i % 4);

        let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);

        sSurface = evalEdit(octant_corner + pixel_offset, sSurface, edits.data[current_unpacked_edit_idx], &current_edit_surface);
    }

    let interpolant : f32 = (f32(edit_culling_count[parent_octree_index]) / f32(5)) * (3.14159265 / 2.0);

    var heatmap_color : vec3f;
    heatmap_color.r = sin(interpolant);
    heatmap_color.g = sin(interpolant * 2.0);
    heatmap_color.b = cos(interpolant);

    textureStore(write_sdf, atlas_tile_coordinate + local_id, vec4f(sSurface.color, sSurface.distance));
    //textureStore(write_sdf, start_writing_pos + local_id - 1, vec4f(heatmap_color, sSurface.distance));

    octant_usage_write[0] = 0;
}
