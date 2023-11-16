#include sdf_functions.wgsl
#include octree_includes.wgsl

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(4) var<storage, read_write> counters : OctreeCounters;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(6) var<storage, read_write> edit_culling_lists: array<u32>;
@group(0) @binding(7) var<storage, read_write> edit_culling_count : array<u32>;
@group(0) @binding(8) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

/*
    Octree Octant indexing
        - 3 bits per each layer, describes 8 octants.
        - With a u32 up to 10 layers -> This gives us indexing in an octree with 1024^3 leaves (10 layers).
        - The base layer (top of the tree) is common to all nodes, so the first layer is skipped in the respresentation.
        - We currently use 7 layers, due or bricks being 8x8x8.
        - The 3 less significant bits represent the octant index of the second layer.
        - The 3 more significant bits represetne the octant index of the leafs´s layer.
            | Leaves 3 bits | .... | 3rd layer 3 bits | 2nd layer 3 bits |
        - That way we can store all the parents in a single word.

    Edit Culling Lists
        - We can only process up to 256 edits in a single compute dispatch.
        - The edit storage is the edits uniform.
        - For each node of the octree, there is a array of 64 u32 indices, that references the 256 edits.
        - Each item in the list is used in intervals of 8 bits, so in one u32 word we store 4 indices.
        - With bit operations we can circunvent the lack of u8 of current webgpu, withou having to use the full u32 for indexing.
        - There are two structures:
            - The Culling Lists: where the actual lists are stored as a single buffer, accessed as a 2D array:
                    edit_culling_lists[in_list_word_index + octree_index * word_list_size] <- word_list_size is 64 here
            - Culling count: a buffer with the same strucutre as the octree, that stores the number of edits in each node
*/

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let level : u32 = atomicLoad(&counters.current_level);

    let id : u32 = group_id.x;

    var parent_level : u32;

    // Edge case: the first level does not have a parent, so it writes the edit list to itself
    if (level == 0) {
        parent_level = level;
    } else {
        parent_level = level - 1;
    }

    let octant_id : u32 = octant_usage_read[id];
    let parent_mask : u32 = u32(pow(2, f32(merge_data.max_octree_depth * 3))) - 1;
    // The parent is indicated in the index, so according to the level, we remove the 3 lower bits, associated to the current octant
    let parent_octant_id : u32 = octant_id & (parent_mask >> (3u * (merge_data.max_octree_depth - parent_level)));

    // In array indexing: in_level_position_of_octant (the octant id) + layer_start_in_array
    // Given a level, we can compute the size of a level with (8^(level-1))/7
    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
    let parent_octree_index : u32 = parent_octant_id + u32((pow(8.0, f32(parent_level)) - 1) / 7);

    var octant_center : vec3f = vec3f(0.0);
    var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

    // Compute the center and the half size of the current octree, in the current level, via iterating the octree index
    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));

        // For each level, select the octant position via the 3 corresponding bits and use the OFFSET_LUT that
        // indicates the relative position of an octant in a layer
        // We offset the octant id depending on the layer that we are, and remove all the trailing bits (if any)
        octant_center += level_half_size * OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x7];
    }

    var sSurface : SurfaceInterval = SurfaceInterval(octree.data[octree_index].octant_center_distance);
    var current_edit_surface : SurfaceInterval;
    var edit_counter : u32 = 0;

    var new_packed_edit_idx : u32 = 0;

    // let cull_distance : f32 = level_half_size * SQRT_3 * 1.5;

    // Check the edits in the parent, and fill its own list with the edits that affect this child
    for (var i : u32 = 0; i < edit_culling_count[parent_octree_index] ; i++) {
        // Accessing a packed indexed edit in the culling list:

        // Get the word index and the word: word_idx = idx / 4
        let current_packed_edit_idx : u32 = edit_culling_lists[i / 4 + parent_octree_index * PACKED_LIST_SIZE];
        //Get the in-word index (inverted for endianess coherency)
        let packed_index : u32 = 3 - (i % 4);

        // Bit-sift for getting the 8 bits that indicate the in-word index:
        //   First, move a 8 bit mask so it coincides with the 8 bits that we want
        //   then apply the mask, and swift the result so the 8 bits are at the start of the word -> unpacked index & profit
        let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);

        let x_range : vec2f = vec2f(octant_center.x - level_half_size * 0.5, octant_center.x + level_half_size * 0.5);
        let y_range : vec2f = vec2f(octant_center.y - level_half_size * 0.5, octant_center.y + level_half_size * 0.5);
        let z_range : vec2f = vec2f(octant_center.z - level_half_size * 0.5, octant_center.z + level_half_size * 0.5);

        let current_edit : Edit = edits.data[current_unpacked_edit_idx];
        sSurface = evalEditInterval(x_range, y_range, z_range, sSurface, current_edit, &current_edit_surface);

        // Check if the edit affects the current voxel, if so adds it to the packed list 
        if (true) {
            // Using the edit counter, sift the edit id to the position in the current word, and adds it
            new_packed_edit_idx = new_packed_edit_idx | (current_unpacked_edit_idx << ((3 - edit_counter % 4) * 8));

            edit_counter++;

            // If the current word is full, we store it, and set it up a new word in new_packed_edit_idx
            if (edit_counter % 4 == 0) {
                edit_culling_lists[(edit_counter - 1) / 4 + octree_index * PACKED_LIST_SIZE] = new_packed_edit_idx;
                new_packed_edit_idx = 0;
                continue;
            }
        } 
        
        // If we are in the last iteration and we have not saved the current packed word, we store it
        if (i == (edit_culling_count[parent_octree_index] - 1)) {
            edit_culling_lists[(edit_counter) / 4 + octree_index * PACKED_LIST_SIZE] = new_packed_edit_idx;
        }
    }

    octree.data[octree_index].octant_center_distance = sSurface.distance;

    edit_culling_count[octree_index] = edit_counter;

    if (level < merge_data.max_octree_depth) {

        // Inside or outside the surface
        if (sSurface.distance.y < 0 || sSurface.distance.x > 0) {
            // Add to the index the childres's octant id, and save it for the next pass
            for (var i : u32 = 0; i < 8; i++) {
                let child_octant_id : u32 = octant_id | (i << (3 * level));
                let child_octree_index : u32 = child_octant_id + u32((pow(8.0, f32(level + 1)) - 1) / 7);
                if ((octree.data[child_octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG) {
                    let prev_counter : u32 = atomicAdd(&counters.atomic_counter, 1);
                    octant_usage_write[prev_counter] = child_octant_id;
                }
            }

            octree.data[octree_index].tile_pointer = 0u;
        }
        // ambiguous, subdivide 
        else {
            // For the 0<->(n-1) passes
            // Increase the number of children from the current level
            let prev_counter : u32 = atomicAdd(&counters.atomic_counter, 8);

            // Add to the index the childres's octant id, and save it for the next pass
            for (var i : u32 = 0; i < 8; i++) {
                octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
            }

            // Mark this node as it has children
            octree.data[octree_index].tile_pointer = FILLED_BRICK_FLAG;
        }
    } else {

        // Inside or outside the surface
        if (sSurface.distance.y < 0 || sSurface.distance.x > 0) {
            if ((FILLED_BRICK_FLAG & octree.data[octree_index].tile_pointer) == FILLED_BRICK_FLAG) {
                let brick_to_delete_idx = atomicAdd(&indirect_brick_removal.brick_removal_counter, 1u);

                let instance_index : u32 = octree.data[octree_index].tile_pointer & 0x3FFFFFFFu;
                indirect_brick_removal.brick_removal_buffer[brick_to_delete_idx] = instance_index;
                octree.data[octree_index].tile_pointer = 0u;
            }

            if (sSurface.distance.y < 0.0) {
                // Mark brick as interior (inside a surface)
                octree.data[octree_index].tile_pointer = INTERIOR_BRICK_FLAG;
            }
        }
        // ambiguous, subdivide 
        else {
            // For the N pass, just send the leaves, to the writing to texture pass
            let prev_counter : u32 = atomicAdd(&counters.atomic_counter, 1);

            // if FILLED_BRICK_FLAG is set, there is already a tile in the octree, if not, we allocate one
            if ((FILLED_BRICK_FLAG & octree.data[octree_index].tile_pointer) != FILLED_BRICK_FLAG) {
                let brick_spot_id = atomicSub(&octree_proxy_data.atlas_empty_bricks_counter, 1u) - 1u;
                let instance_index : u32 = octree_proxy_data.atlas_empty_bricks_buffer[brick_spot_id];
                octree_proxy_data.instance_data[instance_index].position = octant_center;
                octree_proxy_data.instance_data[instance_index].atlas_tile_index = instance_index;
                octree_proxy_data.instance_data[instance_index].octree_parent_id = octree_index;
                octree_proxy_data.instance_data[instance_index].in_use = 1;

                if ((octree.data[octree_index].tile_pointer & INTERIOR_BRICK_FLAG) == INTERIOR_BRICK_FLAG) {
                    octree.data[octree_index].tile_pointer = instance_index | INTERIOR_BRICK_FLAG;
                } else {
                    octree.data[octree_index].tile_pointer = instance_index;
                }
            }

            octant_usage_write[prev_counter] = octree_index;
        }
    }
}
