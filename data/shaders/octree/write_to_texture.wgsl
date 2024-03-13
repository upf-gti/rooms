#include ../math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include material_packing.wgsl
#include ../noise.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(3) var write_sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(6) var<storage, read_write> edit_culling_data: EditCullingData;
@group(0) @binding(7) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;
@group(0) @binding(8) var write_material_sdf: texture_storage_3d<r32uint, read_write>;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

var<workgroup> used_pixels : atomic<u32>;

fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
}

const SDF_MATERIAL_BRICK_DIFFERENCE = u32(MATERIAL_BRICK_SIZE / SDF_BRICK_SIZE);
const PIXEL_WORLD_SIZE_QUARTER = PIXEL_WORLD_SIZE / 4;

const delta_pos_texture = array<vec3u, 8>(
    vec3u(1u,1u,1u),
    vec3u(1u,1u,0u),
    vec3u(1u,0u,1u),
    vec3u(1u,0u,0u),
    vec3u(0u,1u,1u),
    vec3u(0u,1u,0u),
    vec3u(0u,0u,1u),
    vec3u(0u,0u,0u)
);

const delta_pos_world = array<vec3f, 9>(
    vec3f(PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
    vec3f(PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
    vec3f(PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
    vec3f(PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
    vec3f(-PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
    vec3f(-PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
    vec3f(-PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, PIXEL_WORLD_SIZE_QUARTER),
    vec3f(-PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER, -PIXEL_WORLD_SIZE_QUARTER),
    vec3f(0.0, 0.0, 0.0),
);


@compute @workgroup_size(10,10,10)
fn compute(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>)
{
    let id : u32 = group_id.x;
    let octree_leaf_id : u32 = octant_usage_read[id];

    let brick_pointer : u32 = octree.data[octree_leaf_id].tile_pointer;

    // Get the brick index, without the MSb that signals if it has an already initialized brick
    let brick_index : u32 = brick_pointer & OCTREE_TILE_INDEX_MASK;

    let proxy_data : ProxyInstanceData = octree_proxy_data.instance_data[brick_index];

    // Get the 3D atlas coords of the brick, with a stride of the size of the brick
    let atlas_tile_coordinate : vec3u = SDF_BRICK_SIZE * vec3u(proxy_data.atlas_tile_index % BRICK_COUNT,
                                                              (proxy_data.atlas_tile_index / BRICK_COUNT) % BRICK_COUNT,
                                                              proxy_data.atlas_tile_index / (BRICK_COUNT * BRICK_COUNT));

    let level : u32 = atomicLoad(&octree.current_level);
    
    let parent_octree_index : u32 =  proxy_data.octree_parent_id;
    let octant_center : vec3f = proxy_data.position;

    var sSurface : Surface;
    sSurface.distance = 10000.0;
    var debug_surf : vec3f = vec3f(0.0);

    let texture_coordinates : vec3u = atlas_tile_coordinate + local_id;

    var material : Material;
    material.albedo = stroke.material.color.xyz;
    material.roughness = stroke.material.roughness;
    material.metalness = stroke.material.metallic;

    var surface_samples : array<Surface, 9> = array<Surface, 9>();

    let material_coordinates_origin : vec3u = texture_coordinates * SDF_MATERIAL_BRICK_DIFFERENCE + vec3u(1u);

    // If the MSb is setted we load the previous data of brick
    // if not, we set it for the next iteration
    if ((FILLED_BRICK_FLAG & brick_pointer) == FILLED_BRICK_FLAG) {
        let distance : f32 = textureLoad(write_sdf, texture_coordinates).r;

#ifdef DOUBLE_MATERIAL_RES
        // Initialize all the samples with the according material subsamples
        for(var i : u32 = 0u; i < 9u; i++) {
            let texture_pos : vec3u = delta_pos_texture[i];
            let raw_material : u32 = textureLoad(write_material_sdf, material_coordinates_origin + texture_pos).r;

            surface_samples[i] = Surface(unpack_material(u32(raw_material)), distance);
        }  
#else
        // Initialize all the samples with the same material
        sSurface.distance = distance;
        let raw_material : u32 = textureLoad(write_material_sdf, texture_coordinates).r;
        sSurface.material = unpack_material(u32(raw_material));

        surface_samples[0] = sSurface;
        surface_samples[1] = sSurface;
        surface_samples[2] = sSurface;
        surface_samples[3] = sSurface;
        surface_samples[4] = sSurface;
        surface_samples[5] = sSurface;
        surface_samples[6] = sSurface;
        surface_samples[7] = sSurface;
        surface_samples[8] = sSurface;
#endif
    } else {
        // If the brick is not filled, or is an interior brick, we just filled it with a the same material
        if ((INTERIOR_BRICK_FLAG & brick_pointer) == INTERIOR_BRICK_FLAG) {
            sSurface.distance = -100.0;
        }

        // If we are not capturing multiple samples, just take one
#ifdef SSAA_OR_DOUBLE_MATERIAL_RES
        surface_samples[0] = sSurface;
        surface_samples[1] = sSurface;
        surface_samples[2] = sSurface;
        surface_samples[3] = sSurface;
        surface_samples[4] = sSurface;
        surface_samples[5] = sSurface;
        surface_samples[6] = sSurface;
        surface_samples[7] = sSurface;
#endif
        surface_samples[8] = sSurface;
    } 

    // Offset for a 10 pixel wide brick
    let pixel_offset : vec3f = (vec3f(local_id) - f32(SDF_BRICK_SIZE/2.0)) * PIXEL_WORLD_SIZE;

    var result_surface : Surface;
    result_surface.distance = 0.0;

    let pos = octant_center + pixel_offset;
    // Super Sampling iterations

#ifdef SSAA_OR_DOUBLE_MATERIAL_RES
    for(var j : u32 = 0u; j < 9u; j++) {
#else
    let j : u32 = 8u;
#endif
        // Traverse the according edits and evaluate them in the brick
        for (var i : u32 = 0; i < edit_culling_data.edit_culling_count[parent_octree_index]; i++) {
            // Get the packed indices
            let current_packed_edit_idx : u32 = edit_culling_data.edit_culling_lists[i / 4 + parent_octree_index * PACKED_LIST_SIZE];
            let packed_index : u32 = 3 - (i % 4);
            let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);
            let edit = stroke.edits[current_unpacked_edit_idx];

            // Rust example.. ??

            let stroke_color : vec3f = stroke.material.color.xyz;
            
            let noise_color : vec3f = stroke.material.noise_color.xyz;
            let noise_intensity : f32 = stroke.material.noise_params.x;
            let noise_freq : f32 = stroke.material.noise_params.y;
            let noise_octaves : u32 = u32(stroke.material.noise_params.z);

            var noise_value = fbm( pos, vec3f(0.0), noise_freq, noise_intensity, noise_octaves );
            noise_value = clamp(noise_value, 0.0, 1.0);

            material.albedo = mix(stroke_color, stroke_color * noise_color, noise_value);
            material.roughness = mix(stroke.material.roughness, 1.0, noise_value * 1.5);
            material.metalness = mix(stroke.material.metallic, 0.25, noise_value * 1.5);

            surface_samples[j] = evaluate_edit(pos + delta_pos_world[j], stroke.primitive, stroke.operation, stroke.parameters, surface_samples[j], material, edit);
        }

#ifdef SSAA_OR_DOUBLE_MATERIAL_RES

        // Accumulate the sampling results
        result_surface.material = Material_sum_Material(surface_samples[j].material, result_surface.material);
    }

    // Average all the samples
    result_surface.material = Material_mult_by(result_surface.material, 1.0 / 9.0);
    result_surface.distance = surface_samples[8u].distance;
#else
    result_surface = surface_samples[j];
#endif

    // Take note of how many pixels have been written to the current brick
    if (result_surface.distance < MIN_HIT_DIST) {
        atomicAdd(&used_pixels, 1);
    }

    textureStore(write_sdf, texture_coordinates, vec4f(result_surface.distance));

#ifdef DOUBLE_MATERIAL_RES
        // Store all the different subsamples if we have the double material resolution
        for(var i : u32 = 0u; i < 8u; i++) {
            let texture_pos : vec3u = delta_pos_texture[i];

            textureStore(write_material_sdf, material_coordinates_origin + texture_pos, vec4<u32>(pack_material(surface_samples[i].material)));
        }
#else
        // No double of the material resolution, just store the current sample
        textureStore(write_material_sdf, texture_coordinates, vec4<u32>(pack_material(result_surface.material)));
#endif
    
    // Hack, for buffer usage
    octant_usage_write[0] = 0;

    workgroupBarrier();

    if (local_id.x == 0 && local_id.y == 0 && local_id.z == 0) {

        let filled_pixel_count : u32 = atomicLoad(&used_pixels);
        if (filled_pixel_count == 0u) {
            octree_proxy_data.instance_data[brick_index].in_use = 0;
            // Add the brick to the indirect
            let brick_to_delete_idx = atomicAdd(&indirect_brick_removal.brick_removal_counter, 1u);
            indirect_brick_removal.brick_removal_buffer[brick_to_delete_idx] = brick_index;

            octree.data[octree_leaf_id].octant_center_distance = vec2f(10000.0, 10000.0);
            octree.data[octree_leaf_id].tile_pointer = 0u;
        } else {
            // Add "filled" flag and remove "interior" flag
            octree.data[octree_leaf_id].tile_pointer = brick_index | FILLED_BRICK_FLAG;
        }
    }
}
