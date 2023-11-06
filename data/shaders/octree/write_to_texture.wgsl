#include sdf_functions.wgsl
#include octree_includes.wgsl

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var write_sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<storage, read_write> counters : OctreeCounters;
@group(0) @binding(5) var<storage, read_write> proxy_box_position_buffer: array<ProxyInstanceData>;
@group(0) @binding(6) var<storage, read_write> edit_culling_lists: array<u32>;
@group(0) @binding(7) var<storage, read_write> edit_culling_count : array<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

@compute @workgroup_size(10,10,10)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;
    let octree_leaf_id : u32 = octant_usage_read[id];

    let brick_index : u32 = octree.data[octree_leaf_id].tile_pointer;

    let atlas_tile_index : u32 = proxy_box_position_buffer[brick_index & 0x7fffffffu].atlas_tile_index;

    let atlas_tile_coordinate : vec3u = 10 * vec3u(atlas_tile_index % BRICK_COUNT,
                                                  (atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   atlas_tile_index / (BRICK_COUNT * BRICK_COUNT));

    let level : u32 = atomicLoad(&counters.current_level);
    
    let parent_level : u32 = level - 1;

    let parent_octree_index : u32 =  proxy_box_position_buffer[brick_index & 0x7fffffffu].octree_parent_id;
    let octant_center : vec3f = proxy_box_position_buffer[brick_index & 0x7fffffffu].position;

    var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));
    }

    // To start writing at the top corner
    let octant_corner : vec3f = octant_center - vec3f(level_half_size);

    var sSurface : Surface = Surface(vec3f(0.0, 0.0, 0.0), 10000.0);
    var debug_surf : vec3f = vec3f(0.0);

    let texture_coordinates : vec3u = atlas_tile_coordinate + local_id;

    // If the MSb is setted we load the previous data of brick
    // if not, we set it for the next iteration
    if ((0x80000000u & brick_index) == 0x80000000u) {
        let sample : vec4f = textureLoad(write_sdf, texture_coordinates);
        sSurface.distance = sample.r;
        sSurface.color = sample.rgb;
        //debug_surf = vec3f(1.0);
    }

    let pixel_offset : vec3f = (vec3f(local_id) - 1.0) * PIXEL_WORLD_SIZE;

    var current_edit_surface : Surface;

    for (var i : u32 = 0; i < edit_culling_count[parent_octree_index]; i++) {

        let current_packed_edit_idx : u32 = edit_culling_lists[i / 4 + parent_octree_index * PACKED_LIST_SIZE];

        let packed_index : u32 = 3 - (i % 4);

        let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);

        sSurface = evalEdit(octant_corner + pixel_offset, sSurface, edits.data[current_unpacked_edit_idx], &current_edit_surface);
    }

    let interpolant : f32 = (f32( edit_culling_count[parent_octree_index] ) / f32(5)) * (3.14159265 / 2.0);

    var heatmap_color : vec3f;
    heatmap_color.r = sin(interpolant);
    heatmap_color.g = sin(interpolant * 2.0);
    heatmap_color.b = cos(interpolant);

    textureStore(write_sdf, texture_coordinates, vec4f(sSurface.distance));

    //textureStore(write_sdf, texture_coordinates, vec4f(debug_surf.x, debug_surf.y, debug_surf.z, sSurface.distance));
    octant_usage_write[0] = 0;

    workgroupBarrier();

    if (local_id.x == 0 && local_id.y == 0 && local_id.z == 0) {
        octree.data[octree_leaf_id].tile_pointer = brick_index | 0x80000000u;
    }
}
