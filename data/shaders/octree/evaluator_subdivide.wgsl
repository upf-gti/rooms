#include math.wgsl
#include octree_includes.wgsl
#include sdf_commons.wgsl

@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read> stroke_aabbs : AABB_List;
//@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;

@group(2) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(2) @binding(1) var<storage, read_write> octant_usage_write : array<u32>;

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

    //TODO(Juan): this could be LUT, but is it worthy at the expense of BIG SHADER
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
    // TODO(Juan): fix undo redo reeval
    let is_evaluating_preview : bool = ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG);
    let is_evaluating_undo : bool = (octree.evaluation_mode & UNDO_EVAL_FLAG) == UNDO_EVAL_FLAG;

    let eval_aabb_min : vec3f = vec3f(octant_center - level_half_size);
    let eval_aabb_max : vec3f = vec3f(octant_center + level_half_size);

    var current_stroke_interval : vec2f = vec2f(10000.0, 10000.0);
    var surface_interval = vec2f(10000.0, 10000.0);
    var edit_counter : u32 = 0;

    // For adition you can just use the intervals stored on the octree
    // however, for smooth substraction there can be precision issues
    // in the form of some bricks disappearing, and that can be solved by
    // recomputing the context

    var subdivide : bool = false;
    var margin : vec4f = vec4f(0.0);

    subdivide = intersection_AABB_AABB(eval_aabb_min, eval_aabb_max, merge_data.evaluation_AABB_min, merge_data.evaluation_AABB_max);
            
    if (subdivide) {
        // Stroke history culling
        var curr_stroke_count : u32 = 0u;
        var any_stroke_inside : bool = false;
        for(var i : u32 = 0u; i < stroke_aabbs.stroke_count; i++) {
            if (intersection_AABB_AABB(eval_aabb_min, 
                                   eval_aabb_max, 
                                   stroke_aabbs.aabbs[i].min, 
                                   stroke_aabbs.aabbs[i].max)) {
                //curr_stroke_count = curr_stroke_count + 1u;
                any_stroke_inside = true;
            }
        }

        if (any_stroke_inside || is_evaluating_undo) {
            // Increase the number of children from the current level
            let prev_counter : u32 = atomicAdd(&octree.atomic_counter, 8);

            // Add to the index the childres's octant id, and save it for the next pass
            for (var i : u32 = 0; i < 8; i++) {
                octant_usage_write[prev_counter + i] = octant_id | (i << (3 * level));
            }
        }
    }
}