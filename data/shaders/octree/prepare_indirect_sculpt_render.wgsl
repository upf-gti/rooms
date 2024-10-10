#include octree_includes.wgsl

struct SculptInstances {
    instance_count : u32,
    model_count : u32,
    indices : array<u32>
};

@group(0) @binding(0) var<storage, read_write> sculpt_instance_count : SculptInstances;

@group(1) @binding(0) var<storage, read_write> octree : Octree_NonAtomic;
@group(1) @binding(1) var<storage, read_write> brick_index_buffer : array<u32>;
@group(1) @binding(2) var<storage, read_write> sculpt_indirect : SculptIndirectCall_NonAtomic;

/**
    Este shader setea los indirect buffers para el render
*/

@compute @workgroup_size(1u, 1u, 1u)
fn compute(@builtin(workgroup_id) id: vec3<u32>)
{
    // TODO: create a bindgroup just for the indirect
    let pad : u32 = octree.current_level;
    let pad1 : u32 = brick_index_buffer[0u];

    let curr_instance_idx : u32 = sculpt_instance_count.instance_count;
    sculpt_instance_count.instance_count = sculpt_instance_count.instance_count + 1u;

    let curr_sculpt_model_count : u32 = sculpt_instance_count.indices[curr_instance_idx];

    sculpt_indirect.instance_count = sculpt_indirect.brick_count * curr_sculpt_model_count;

    // Set the starting idx for the model buffer
    sculpt_indirect.starting_model_idx = sculpt_instance_count.model_count;

    sculpt_instance_count.model_count = sculpt_instance_count.model_count + curr_sculpt_model_count;
}