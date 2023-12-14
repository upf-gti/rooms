#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(1) var<storage, read_write> octant_usage_write_0 : array<u32>;
@group(0) @binding(2) var<storage, read_write> octant_usage_write_1: array<u32>;
@group(0) @binding(4) var<storage, read_write> state : OctreeState;
@group(0) @binding(6) var<storage, read_write> edit_culling_data: EditCullingData;
@group(0) @binding(8) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    indirect_buffer = vec3u(1, 1, 1);

    octant_usage_write_0[0] = 0;
    octant_usage_write_1[0] = 0;

    atomicStore(&state.current_level, 0);
    atomicStore(&state.atomic_counter, 0);
    atomicStore(&state.proxy_instance_counter, 0);

    atomicStore(&indirect_brick_removal.brick_removal_counter, 0);

    indirect_brick_removal.indirect_padding = vec3u(1, 1, 1);
    
    let rounded_size : u32 = stroke.edit_count + (4 - stroke.edit_count % 4);

    for (var i : u32 = 0; i < rounded_size; i += 4) {

        var packed_value : u32 = 0;

        packed_value |= (i + 3);
        packed_value |= (i + 2) << 8;
        packed_value |= (i + 1) << 16;
        packed_value |= (i + 0) << 24;
        
        edit_culling_data.edit_culling_lists[i / 4] = packed_value;
    }

    edit_culling_data.edit_culling_count[0] = stroke.edit_count; 
}