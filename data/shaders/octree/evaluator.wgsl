#include ../math.wgsl
#include sdf_functions.wgsl
#include octree_includes.wgsl
#include sdf_interval_functions.wgsl

@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(5) var<storage, read_write> octree_proxy_data: OctreeProxyInstances;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory; 
@group(0) @binding(8) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

@group(3) @binding(0) var<storage, read_write> preview_data : PreviewData;

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

    Inside bricks:
        - For a correct representation, we would need to keep bricks inside the scuplture, in order for the substraction to work.
        - However, this seemed like a wastefull memory usage.
        - In order to prevent this, we found that marking in the octree the leaves that are inside the SDF as INTERIOR_BRICKS, and
          when substracting and needing to fill the inside bricks, we just set the default distance of 1000.0 of the default SDF to
          -1000.0 according to the flag.
        - This proved sufficient for this usecase. 

    Interval Arithmetic
        - In order to subdivide the octree and cull the edits in an accurate manner, we use a different version of the SDF function,
          that are built using interval arithmetic. You input a range in the function and they return the lowest and the biggest values
          that the function (in this case, the distance) can have in that interval.
        - We can use this in order to determine if there is surface in a particular area of our evaluation area: If the lowest value is less
          than 0, we know that in this area the distance is negative, and that means that there is AT LEAST a bit of surface in that block; 
          and we can subdivide in that area.
        
    SDF Exactness & boundness:
        - However when evaluating we found issues with SDF exactness. Some SDF are exact, and some operations also returns an exact SDF.
        - But, for example, Substraction results in a bound SDF, which means in practice that the propagation of the change in the distance
          is not correct; being only correct near where the operation or de SDF was placed.
        - This is fine for rendering, since the distances are "more correct" the closer you are, but in subdivision, this leads to incorrect
          culling. This is because due the nature of interval arithmetic, having an "incorrectly propagated" distance does not really change
          the minum and maximun interval in an area; unless it is a really big operation, slightly less bigger than the evaluationg surface.
        - In order to counter-meassure this, we subdivide using the incomming edits as exact Union operations, whose distances propagate 
          correctly (called new_edits).
        - Subdivision happens only on the area that is inside or along the surface of the incoming edits.
        - On the last layer, where we decide which brick do we write, delete or re-write, we decide based on the type of edit:
            - If its an union, we delete all bricks iside the resulting SDF, end write/re-evaluate the bricks on the surface of the resulting SDF.
            - If its a substraction, we delete all bricks inside the new_edits, and only create/re-evaluate bricks on the intersection of the 
              new_edits and the resulting SDF.


    TODO:
     - Redo edit culling with interval arithmetic.
     - Union operation interior brick check
*/

fn is_inside_AABB(point : vec3f, aabb_min : vec3f, aabb_max : vec3f) -> bool {
    return (aabb_min.x <= point.x && point.x <= aabb_max.x) && (aabb_min.y <= point.y && point.y <= aabb_max.y) && (aabb_min.z <= point.z && point.z <= aabb_max.z);
}

fn fully_inside_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return is_inside_AABB(b1_min, b2_min, b2_max) && is_inside_AABB(b1_max, b2_min, b2_max);
}

fn intersection_AABB_AABB(b1_min : vec3f, b1_max : vec3f, b2_min : vec3f, b2_max : vec3f) -> bool {
    return (b1_min.x <= b2_max.x && b1_min.y <= b2_max.y && b1_min.z <= b2_max.z) && (b1_max.x >= b2_min.x && b1_max.y >= b2_min.y && b1_max.z >= b2_min.z);
}

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u, @builtin(num_workgroups) workgroup_size : vec3u) 
{
    let level : u32 = atomicLoad(&octree.current_level);

    let id : u32 = group_id.x;

    var parent_level : u32;

    // Edge case: the first level does not have a parent, so it writes the edit list to itself
    if (level == 0) {
        parent_level = level;
    } else {
        parent_level = level - 1;
    }

    let octant_id : u32 = octant_usage_read[id];
    let parent_mask : u32 = u32(pow(2, f32(OCTREE_DEPTH * 3))) - 1;
    // The parent is indicated in the index, so according to the level, we remove the 3 lower bits, associated to the current octant
    let parent_octant_id : u32 = octant_id & (parent_mask >> (3u * (OCTREE_DEPTH - parent_level)));

    // In array indexing: in_level_position_of_octant (the octant id) + layer_start_in_array
    // Given a level, we can compute the size of a level with (8^(level-1))/7
    let octree_index : u32 = octant_id + u32((pow(8.0, f32(level)) - 1) / 7);
    let parent_octree_index : u32 = parent_octant_id + u32((pow(8.0, f32(parent_level)) - 1) / 7);

    var octant_center : vec3f = vec3f(0.0);
    var level_half_size : f32 = 0.5 * SCULPT_MAX_SIZE;

    // Compute the center and the half size of the current octree, in the current level, via iterating the octree index
    for (var i : u32 = 1; i <= level; i++) {
        level_half_size = SCULPT_MAX_SIZE / pow(2.0, f32(i + 1));

        // For each level, select the octant position via the 3 corresponding bits and use the OCTREE_CHILD_OFFSET_LUT that
        // indicates the relative position of an octant in a layer
        // We offset the octant id depending on the layer that we are, and remove all the trailing bits (if any)
        octant_center += level_half_size * OCTREE_CHILD_OFFSET_LUT[(octant_id >> (3 * (i - 1))) & 0x7];
    }

    // Note: the preview evaluation only happens at the end of the frame, so it must wait for
    //       any reevaluation and evaluation
    let is_reevaluation : bool = (octree.evaluation_mode & STROKE_CLEAN_BEFORE_EVAL_FLAG) == STROKE_CLEAN_BEFORE_EVAL_FLAG;
    let is_evaluating_preview : bool = !is_reevaluation && ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG);
    
    // Select the input stroke, or the preview stroke, depending on the mode
    var current_stroke : Stroke = stroke;
    current_stroke = preview_data.preview_stroke;
    //  if (is_evaluating_preview) {
         
    // } else {
         
    //  }
     current_stroke = stroke;

    let is_smooth_paint : bool = current_stroke.operation == OP_SMOOTH_PAINT;
    let is_paint : bool = current_stroke.operation == OP_PAINT; 
    
    let is_union : bool = current_stroke.operation == OP_UNION;
    let is_smooth_union : bool = current_stroke.operation == OP_SMOOTH_UNION;
    let is_any_union : bool = is_union || is_smooth_union;

    let is_substract : bool = current_stroke.operation == OP_SUBSTRACTION;
    let is_smooth_substract : bool = current_stroke.operation == OP_SMOOTH_SUBSTRACTION;
    let is_any_substract : bool = is_substract || is_smooth_substract;

    let is_any_smooth : bool = is_smooth_substract || is_smooth_paint || is_smooth_union;
    
    let octant_min : vec3f = octant_center - vec3f(level_half_size);
    let octant_max : vec3f = octant_center + vec3f(level_half_size);

    let is_in_reevaluation_zone : bool = intersection_AABB_AABB(merge_data.reevaluation_AABB_min, merge_data.reevaluation_AABB_max, octant_min, octant_max);

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = (octree.data[octree_index].octant_center_distance);
    var edit_counter : u32 = 0;

    var new_packed_edit_idx : u32 = 0;

    var edit_cutoff_distance : f32 = MIN_HIT_DIST;


    // Base evaluation range
    let x_range : vec2f = vec2f(octant_center.x - level_half_size, octant_center.x + level_half_size);
    let y_range : vec2f = vec2f(octant_center.y - level_half_size, octant_center.y + level_half_size);
    let z_range : vec2f = vec2f(octant_center.z - level_half_size, octant_center.z + level_half_size);

    // Check the edits in the parent, and fill its own list with the edits that affect this child
    for (var i : u32 = 0; i < current_stroke.edit_count; i++) {
        // Accessing a packed indexed edit in the culling list:

        // Get the word index and the word: word_idx = idx / 4
        // let current_packed_edit_idx : u32 = edit_culling_data.edit_culling_lists[i / 4 + parent_octree_index * PACKED_LIST_SIZE];
        // //Get the in-word index (inverted for endianess coherency)
        // let packed_index : u32 = 3 - (i % 4);

        // // Bit-sift for getting the 8 bits that indicate the in-word index:
        // //   First, move a 8 bit mask so it coincides with the 8 bits that we want
        // //   then apply the mask, and swift the result so the 8 bits are at the start of the word -> unpacked index & profit
        // let current_unpacked_edit_idx : u32 = (current_packed_edit_idx & (0xFFu << (packed_index * 8u))) >> (packed_index * 8u);

        var current_edit : Edit = current_stroke.edits[i];
        var edit_interval : vec2f;

        surface_interval = eval_edit_interval(x_range, y_range, z_range, current_stroke.primitive, current_stroke.operation, current_stroke.parameters, surface_interval, current_edit, &edit_interval);

        current_edit.dimensions += vec4f(current_stroke.parameters.w, 0.0, 0.0, 0.0);  

        current_stroke_interval = eval_edit_interval(x_range, y_range, z_range, current_stroke.primitive, OP_UNION, current_stroke.parameters, current_stroke_interval, current_edit, &edit_interval);
    }

    let global_sculpture_inside : bool = surface_interval.y < 0.0;
    let global_sculpture_outside : bool = surface_interval.x > 0.0;
    let global_sculpture_in_surface : bool = surface_interval.x < 0.0 && surface_interval.y > 0.0;

    let current_stroke_inside : bool = current_stroke_interval.y < 0.0;
    let current_stroke_outside : bool = current_stroke_interval.x > 0.0;
    let current_stroke_in_surface : bool = current_stroke_interval.x < 0.0 && current_stroke_interval.y > 0.0;

    let is_current_brick_filled : bool = (octree.data[octree_index].tile_pointer & FILLED_BRICK_FLAG) == FILLED_BRICK_FLAG;
    let is_interior_brick : bool = (octree.data[octree_index].tile_pointer & INTERIOR_BRICK_FLAG) == INTERIOR_BRICK_FLAG;

    if (!is_evaluating_preview) {
        octree.data[octree_index].octant_center_distance = surface_interval;
    }
    
    //edit_culling_data.edit_culling_count[octree_index] = 0u;

    
    if (level < OCTREE_DEPTH) {
        // Broad culling using only the incomming stroke
        if (current_stroke_interval.x < 0.0) {
            // Subdivide
            // Increase the number of children from the current level
            let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 8);

            // Add to the index the childres's octant id, and save it for the next pass
            for (var i : u32 = 0; i < 8; i++) {
                octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
            }
    }
    } else {
        // Last level, more coarse culling
        let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 1);
        let brick_spot_id = atomicSub(&octree_proxy_data.atlas_empty_bricks_counter, 1u) - 1u;
                    let instance_index : u32 = octree_proxy_data.atlas_empty_bricks_buffer[brick_spot_id];
                    octree_proxy_data.instance_data[instance_index].position = octant_center;
                    octree_proxy_data.instance_data[instance_index].atlas_tile_index = instance_index;
                    octree_proxy_data.instance_data[instance_index].octree_parent_id = octree_index;
                    octree_proxy_data.instance_data[instance_index].in_use = BRICK_IN_USE_FLAG;

                    // If it was interior, we mark it as such
                    // TODO: This cannot happen now right??
                    if (is_interior_brick) {
                        octree.data[octree_index].tile_pointer = instance_index | INTERIOR_BRICK_FLAG;
                        octree.data[octree_index].octant_center_distance = vec2f(-10000.0, -10000.0);
                    } else {
                        octree.data[octree_index].tile_pointer = instance_index;
                    }
                    octant_usage_write[prev_counter] = octree_index;

                }
                
    
     let z : u32 =  stroke_history.count;
    let v : u32 =  octree_proxy_data.instance_data[0].in_use;
    let j : u32 = indirect_brick_removal.indirect_padding.x;
}