#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var write_sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<storage, read_write> counters : OctreeCounters;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(6) var<storage, read_write> edit_culling_lists: array<u32>;
@group(0) @binding(7) var<storage, read_write> edit_culling_count : array<u32>;
@group(0) @binding(8) var write_material_sdf: texture_storage_3d<r32uint, read_write>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

var<workgroup> used_pixels : atomic<u32>;

@compute @workgroup_size(10,10,10)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;
    let octree_leaf_id : u32 = octant_usage_read[id];

    let brick_pointer : u32 = octree.data[octree_leaf_id].tile_pointer;

    // Get the brick index, without the MSb that signals if it has an already initialized brick
    let brick_index : u32 = brick_pointer & 0x3FFFFFFFu;

    let proxy_data : ProxyInstanceData = octree_proxy_data.instance_data[brick_index];

    // Get the 3D atlas coords of the brick, with a stride of 10 (the size of the brick)
    let atlas_tile_coordinate : vec3u = 10 * vec3u(proxy_data.atlas_tile_index % BRICK_COUNT,
                                                  (proxy_data.atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                   proxy_data.atlas_tile_index / (BRICK_COUNT * BRICK_COUNT));

    let level : u32 = atomicLoad(&counters.current_level);
    
    let parent_octree_index : u32 =  proxy_data.octree_parent_id;
    let octant_center : vec3f = proxy_data.position;

    let level_half_size : f32 = SCULPT_MAX_SIZE / pow(2.0, f32(level + 1));

    var sSurface : Surface = Surface(vec3f(0.0, 0.0, 0.0), 10000.0);
    var debug_surf : vec3f = vec3f(0.0);

    let texture_coordinates : vec3u = atlas_tile_coordinate + local_id;

    // If the MSb is setted we load the previous data of brick
    // if not, we set it for the next iteration
    if ((FILLED_BRICK_FLAG & brick_pointer) == FILLED_BRICK_FLAG) {
        let sample : vec4f = textureLoad(write_sdf, texture_coordinates);
        sSurface.distance = sample.r;
        let raw_color : vec4<u32> = textureLoad(write_material_sdf, texture_coordinates);

        let material : Material = unpack_material(u32(raw_color.r));
        sSurface.color = material.albedo;
        //debug_surf = vec3f(1.0);
    } else
    if ((INTERIOR_BRICK_FLAG & brick_pointer) == INTERIOR_BRICK_FLAG) {
        sSurface.distance = -100.0;
    }

    // Offset for a 10 pixel wide brick
    let pixel_offset : vec3f = (vec3f(local_id) - 4.5) * PIXEL_WORLD_SIZE;

    var current_edit_surface : Surface;

    // Traverse the according edits and evaluate them in the brick
    for (var i : u32 = 0; i < edit_culling_count[parent_octree_index]; i++) {
        // Get the packed indices
        let current_packed_edit_idx : u32 = edit_culling_lists[i / 4 + parent_octree_index * PACKED_LIST_SIZE];
        let packed_index : u32 = 3 - (i % 4);
        let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);

        sSurface = evalEdit(octant_center + pixel_offset, sSurface, edits.data[current_unpacked_edit_idx], &current_edit_surface);
    }

    if (sSurface.distance < MIN_HIT_DIST) {
        atomicAdd(&used_pixels, 1);
    }

    // Heatmap Edit debugging
    let interpolant : f32 = (f32( edit_culling_count[parent_octree_index] ) / f32(5)) * (3.14159265 / 2.0);
    var heatmap_color : vec3f;
    heatmap_color.r = sin(interpolant);
    heatmap_color.g = sin(interpolant * 2.0);
    heatmap_color.b = cos(interpolant);

    textureStore(write_sdf, texture_coordinates, vec4f(sSurface.distance));

    var material : Material;
    material.albedo = sSurface.color;
    material.roughness = 0.7;
    material.metalness = 0.2;
    textureStore(write_material_sdf, texture_coordinates, vec4<u32>((pack_material(material))));

    //textureStore(write_sdf, texture_coordinates, vec4f(debug_surf.x, debug_surf.y, debug_surf.z, sSurface.distance));
    // Hack, for buffer usage
    octant_usage_write[0] = 0;

    workgroupBarrier();

    if (local_id.x == 0 && local_id.y == 0 && local_id.z == 0) {

        let filled_pixel_count : u32 = atomicLoad(&used_pixels);
        if (filled_pixel_count > 0u && filled_pixel_count < 1000u) {
            octree_proxy_data.instance_data[brick_index].in_use = 1;
        } 
        // else {
        //     octree_proxy_data.instance_data[brick_index].in_use = 0;
        // }

        // Add "filled" flag and remove "interior" flag
        octree.data[octree_leaf_id].tile_pointer = brick_index | FILLED_BRICK_FLAG;
    }
}
