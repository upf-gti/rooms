#include sdf_functions.wgsl

struct MergeData {
    edits_to_process : u32,
    dummy0           : f32,
    dummy1           : f32,
    dummy2           : f32,
};

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<uniform> merge_data : MergeData;
@group(0) @binding(2) var<storage, read_write> sdf_data : SdfData;

const smooth_factor = 0.6;

fn evalSdf(position : vec3f) -> Surface
{
    var sSurface : Surface = sdf_data.data[u32(position.x + position.y * 512 + position.z * 512 * 512)];

    for (var i : u32 = 0; i < merge_data.edits_to_process; i++) {

        let edit : Edit = edits.data[i];

        var pSurface : Surface;

        let offsetPosition : vec3f = edit.position;

        switch (edit.primitive) {
            case SD_SPHERE: {
                pSurface = sdSphere(position, offsetPosition, edit.radius, edit.color);
                break;
            }
            case SD_BOX: {
                pSurface = sdBox(position, offsetPosition, edit.size, edit.radius, edit.color);
                break;
            }
            // case SD_ELLIPSOID:
            //     pSurface = sdEllipsoid(position, offsetPosition, edit.size, edit.color);
            //     break;
            // case SD_CONE:
            //     pSurface = sdCone(position, offsetPosition, edit.size.xy, edit.size.z, edit.color);
            //     break;
            // case SD_PYRAMID:
            //     pSurface = sdPyramid(position, offsetPosition, edit.size.x, edit.radius, edit.color);
            //     break;
            // case SD_CYLINDER:
            //     pSurface = sdCylinder(position, offsetPosition, offsetPosition + vec3(0.0, 5.0, 0.0), edit.size.x, edit.radius, edit.color);
            //     break;
            // case SD_CAPSULE:
            //     pSurface = sdCapsule(position, offsetPosition, offsetPosition + vec3(2.0, 5.0, 0.0), edit.radius, edit.color);
            //     break;
            // case SD_TORUS:
            //     pSurface = sdTorus(position, offsetPosition, edit.size.xy, edit.color);
            //     break;
            // case SD_CAPPED_TORUS:
            //     pSurface = sdCappedTorus(position, offsetPosition, edit.size.x, edit.size.y, edit.size.z, edit.color);
            //     break;
            default: {
                break;
            }
        }

        switch (edit.operation) {
            case OP_UNION: {
                sSurface = opUnion(sSurface, pSurface);
                break;
            }
            case OP_SUBSTRACTION:{
                sSurface = opSubtraction(sSurface, pSurface);
                break;
            }
            case OP_INTERSECTION: {
                sSurface = opIntersection(sSurface, pSurface);
                break;
            }
            case OP_PAINT: {
                sSurface = opPaint(sSurface, pSurface, edit.color);
                break;
            }
            case OP_SMOOTH_UNION: {
                sSurface = opSmoothUnion(sSurface, pSurface, smooth_factor);
                break;
            }
            case OP_SMOOTH_SUBSTRACTION: {
                sSurface = opSmoothSubtraction(sSurface, pSurface, smooth_factor);
                break;
            }
            case OP_SMOOTH_INTERSECTION: {
                sSurface = opSmoothIntersection(sSurface, pSurface, smooth_factor);
                break;
            }
            case OP_SMOOTH_PAINT: {
                sSurface = opSmoothPaint(sSurface, pSurface, edit.color, smooth_factor);
                break;
            }
            default: {
                break;
            }
        }
     }

    return sSurface;
}

@compute @workgroup_size(8, 8, 8)

fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let position : vec3i = vec3i(id);
    sdf_data.data[position.x + position.y * 512 + position.z * 512 * 512] = evalSdf(vec3f(position));
}
