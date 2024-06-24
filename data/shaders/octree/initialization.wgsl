#include octree_includes.wgsl

@group(0) @binding(1) var<storage, read_write> octant_usage_write_0 : array<u32>;
@group(0) @binding(2) var<storage, read_write> octant_usage_write_1: array<u32>;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers_ReadOnly;
@group(0) @binding(6) var<storage, read> stroke_history : StrokeHistory;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers_ReadOnly;
@group(0) @binding(9) var<storage, read_write> stroke_culling : array<u32>;

@group(1) @binding(0) var<storage, read_write> octree : Octree;



/**
    Este shader prepara los buffers para la evaluacion de strokes, que vienen de la CPU.
    La funcion principal es limpiar los indirect buffers para el resto de llamadas.
    Como el compute del sistema de evaluador es GPU driven, es fundamental que esto se
    haga por cada vez que evaluamos un nuevo stroke.
*/

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    // Clean the structs for the preview

    octant_usage_write_0[0] = 0;
    octant_usage_write_1[0] = 0;

    // TODO(Juan): Noe snecessairo ahora estos atomicstroe sustituir por store normal
    atomicStore(&octree.current_level, 0);
    atomicStore(&octree.atomic_counter, 0);

    brick_buffers.preview_instance_counter = 0u;
    indirect_buffers.preview_instance_count = 0u;

    if ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG) {
        
    } else {
        indirect_buffers.brick_removal_counter = 0u;
        brick_buffers.brick_removal_counter = 0u;
        indirect_buffers.brick_instance_count = 0u;
        brick_buffers.brick_instance_counter = 0u;

        // Store the culling data of the first level
        for(var i = 0u; i < stroke_history.count; i++){
            stroke_culling[i] = culling_get_culling_data(i, 0, stroke_history.strokes[i].edit_count);
        }
        octree.data[0].stroke_count = stroke_history.count;
    }

    indirect_buffers.evaluator_subdivision_counter = 1u;
}