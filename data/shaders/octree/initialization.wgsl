#include octree_includes.wgsl

@group(0) @binding(0) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(1) var<storage, read_write> octant_usage_write_0 : array<u32>;
@group(0) @binding(2) var<storage, read_write> octant_usage_write_1: array<u32>;
@group(0) @binding(4) var<storage, read_write> octree : Octree;
@group(0) @binding(6) var<storage, read_write> edit_culling_data: EditCullingData;
@group(0) @binding(8) var<storage, read_write> indirect_brick_removal : IndirectBrickRemoval;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;

@group(2) @binding(0) var<storage, read_write> preview_data : PreviewData;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    // Clean the structs for the preview
    if ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG) {
        atomicStore(&preview_data.instance_count, 0u);

        indirect_buffer = vec3u(1, 1, 1);

        octant_usage_write_0[0] = 0;
        octant_usage_write_1[0] = 0;

        atomicStore(&octree.current_level, 0);
        atomicStore(&octree.atomic_counter, 0);

        let rounded_size : u32 = preview_data.preview_stroke.edit_count + (4 - preview_data.preview_stroke.edit_count % 4);

        for (var i : u32 = 0; i < rounded_size; i += 4) {

            var packed_value : u32 = 0;

            packed_value |= (i + 3);
            packed_value |= (i + 2) << 8;
            packed_value |= (i + 1) << 16;
            packed_value |= (i + 0) << 24;
            
            edit_culling_data.edit_culling_lists[i / 4] = packed_value;
        }
        edit_culling_data.edit_culling_count[0] = preview_data.preview_stroke.edit_count; 
    } else {
        indirect_buffer = vec3u(1, 1, 1);

        octant_usage_write_0[0] = 0;
        octant_usage_write_1[0] = 0;

        atomicStore(&octree.current_level, 0);
        atomicStore(&octree.atomic_counter, 0);
        atomicStore(&octree.proxy_instance_counter, 0);

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
}