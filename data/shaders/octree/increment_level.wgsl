#include octree_includes.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(5) var<storage, read> brick_buffers: BrickBuffers_ReadOnly;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers_ReadOnly;

/**
    Este shader se llama despues de cada pasada de evaluator, y su fin es configurar el
    indirect buffer para llamar al siguiente shader e incrementar el nivel actual
    de la subdivision.
    En el caso de que esta llamada sea la ultima (estamos en el ultimo piso 
    de la subdivision) en caso de que sea necesario desactivamos la flag de
    calcular el preview.
*/

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    // TODO(Juan): remove this atomics, does not make sense now
    let num_dispatches : u32 = atomicLoad(&octree.atomic_counter);

    let level : u32 = atomicAdd(&octree.current_level, 1);

    atomicStore(&octree.atomic_counter, 0u);

    // Update the indirect buffer
    indirect_buffers.evaluator_subdivision_counter = num_dispatches;
    indirect_buffers.brick_instance_count = brick_buffers.brick_instance_counter;
    indirect_buffers.brick_removal_counter = brick_buffers.brick_removal_counter;

    if (level == OCTREE_DEPTH) {
        // If we evaluated the preview in the prev subdivision pass, we set it back.
        if ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG) {
            
            indirect_buffers.preview_instance_count = brick_buffers.preview_instance_counter;
        } else {
            octree.evaluation_mode = EVALUATE_PREVIEW_STROKE_FLAG;
           // octree.evaluation_mode = EVALUATE_PREVIEW_STROKE_FLAG;
            indirect_buffers.preview_instance_count = brick_buffers.preview_instance_counter;
        }
    }
}
