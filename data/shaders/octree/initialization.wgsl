#include octree_includes.wgsl

@group(0) @binding(1) var<storage, read_write> octant_usage_write_0 : array<u32>;
@group(0) @binding(2) var<storage, read_write> octant_usage_write_1: array<u32>;
@group(0) @binding(4) var<storage, read_write> octree : Octree;
@group(0) @binding(5) var<storage, read_write> brick_buffers: BrickBuffers_ReadOnly;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers_ReadOnly;

#dynamic @group(1) @binding(0) var<storage, read> stroke : Stroke;


/**
    Este shader prepara los buffers para la evaluacion de strokes, que vienen de la CPU.
    La funcion principal es limpiar los indirect buffers para el resto de llamadas.
    Como el compute del sistema de evaluador es GPU driven, es fundamental que esto se
    haga por cada vez que evaluamos un nuevo stroke.
*/

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(workgroup_id) group_id: vec3u) 
{
    //let tmp = edit_culling_data.edit_culling_lists[0];
    let tmp2 = stroke.edit_count;
    // Clean the structs for the preview

    octant_usage_write_0[0] = 0;
    octant_usage_write_1[0] = 0;

    // TODO(Juan): Noe snecessairo ahora estos atomicstroe sustituir por store normal
    atomicStore(&octree.current_level, 0);
    atomicStore(&octree.atomic_counter, 0);

    if ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG) {
        brick_buffers.preview_instance_counter = 0u;
        indirect_buffers.preview_instance_count = 0u;
    } else {
        indirect_buffers.brick_removal_counter = 0u;
        brick_buffers.brick_removal_counter = 0u;
        indirect_buffers.brick_instance_count = 0u;
        brick_buffers.brick_instance_counter = 0u;
    }

    indirect_buffers.evaluator_subdivision_counter = 1u;
}