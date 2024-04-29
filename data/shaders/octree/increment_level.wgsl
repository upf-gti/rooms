#include octree_includes.wgsl

@group(0) @binding(2) var<storage, read_write> octree : Octree;
@group(0) @binding(8) var<storage, read_write> indirect_buffers : IndirectBuffers;

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
    let num_dispatches : u32 = atomicLoad(&octree.atomic_counter);

    indirect_buffers.evaluator_subdivision_counter = num_dispatches;

    let level : u32 = atomicAdd(&octree.current_level, 1);

    atomicStore(&octree.atomic_counter, 0u);

    if (level >= OCTREE_DEPTH) {
        // We remove the reevaluation flag, if its setted, after finishing teh first pass
        // of the whole octree.
        octree.evaluation_mode &= ~STROKE_CLEAN_BEFORE_EVAL_FLAG;

        // If we evaluated the preview in the prev subdivision pass, we set it back.
        if ((octree.evaluation_mode & EVALUATE_PREVIEW_STROKE_FLAG) == EVALUATE_PREVIEW_STROKE_FLAG) {
            octree.evaluation_mode = 0;
        }
    }
}
