#include sdf_functions.wgsl

fn eval_preview_stroke_sphere_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCutSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
        }
    }
    
    return result_surface;
}

fn eval_preview_stroke_sphere_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCutSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }
    }

    return result_surface;
}

fn eval_preview_stroke_sphere_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCutSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
        }
    }
    
    return result_surface;
}


/*
______           
| ___ \          
| |_/ / _____  __
| ___ \/ _ \ \/ /
| |_/ / (_) >  < 
\____/ \___/_/\_\
*/
fn eval_preview_stroke_box_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_box_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_box_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
    }

    return result_surface;
}

// /*
//  _____                       _      
// /  __ \                     | |     
// | /  \/ __ _ _ __  ___ _   _| | ___ 
// | |    / _` | '_ \/ __| | | | |/ _ \
// | \__/\ (_| | |_) \__ \ |_| | |  __/
//  \____/\__,_| .__/|___/\__,_|_|\___|
//             | |                     
//             |_|
// */

fn eval_preview_stroke_capsule_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_capsule_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_capsule_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
    }

    return result_surface;
}

// /*
//  _____                  
// /  __ \                 
// | /  \/ ___  _ __   ___ 
// | |    / _ \| '_ \ / _ \
// | \__/\ (_) | | | |  __/
//  \____/\___/|_| |_|\___|
// */

fn eval_preview_stroke_cone_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_cone_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_cone_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
    }

    return result_surface;
}


// /*
//  _____       _ _           _           
// /  __ \     | (_)         | |          
// | /  \/_   _| |_ _ __   __| | ___ _ __ 
// | |   | | | | | | '_ \ / _` |/ _ \ '__|
// | \__/\ |_| | | | | | | (_| |  __/ |   
//  \____/\__, |_|_|_| |_|\__,_|\___|_|   
//         __/ |                          
//        |___/
// */


fn eval_preview_stroke_cylinder_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_cylinder_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_cylinder_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
        
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
    }

    return result_surface;
}

// /*
//  _____                    
// |_   _|                   
//   | | ___  _ __ _   _ ___ 
//   | |/ _ \| '__| | | / __|
//   | | (_) | |  | |_| \__ \
//   \_/\___/|_|   \__,_|___/
// */

fn eval_preview_stroke_torus_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
        }
    }
    
    return result_surface;
}

fn eval_preview_stroke_torus_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }
    }

    return result_surface;
}

fn eval_preview_stroke_torus_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
        }
    } else {
        for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = preview_stroke.edit_list[i];
            tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
        }
    }
    
    return result_surface;
}

/*
 _   _           _           
| | | |         (_)          
| | | | ___  ___ _  ___ __ _ 
| | | |/ _ \/ __| |/ __/ _` |
\ \_/ /  __/\__ \ | (_| (_| |
 \___/ \___||___/_|\___\__,_|
*/


fn eval_preview_stroke_vesica_union( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;       
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_vesica_substraction( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;       
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_preview_stroke_vesica_paint( position : vec3f, current_surface : Surface, edit_starting_idx : u32, edit_count : u32 ) -> Surface
{
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let curr_stroke : ptr<storage, Stroke> = &preview_stroke.stroke;   
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;
    let material : SdfMaterial = SdfMaterial(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let color_blend_override_factor : f32 = parameters.z;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = preview_stroke.edit_list[i];
        tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, color_blend_override_factor, material, smooth_factor);
    }

    return result_surface;
}

/*
 _____ _             _          _____           _             _   _             
/  ___| |           | |        |  ___|         | |           | | (_)            
\ `--.| |_ _ __ ___ | | _____  | |____   ____ _| |_   _  __ _| |_ _  ___  _ __  
 `--. \ __| '__/ _ \| |/ / _ \ |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
/\__/ / |_| | | (_) |   <  __/ | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |
\____/ \__|_|  \___/|_|\_\___| \____/ \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
*/

fn evaluate_preview_stroke( position: vec3f, current_surface : Surface, edit_starting_index : u32, edit_count : u32) -> Surface {
    let stroke : ptr<storage, Stroke> = &preview_stroke.stroke;
    let stroke_operation : u32 = (*stroke).operation;
    let stroke_primitive : u32 = (*stroke).primitive;

    let curr_stroke_code : u32 = stroke_primitive | (stroke_operation << 4u);

    var result_surface : Surface = current_surface;

    switch(curr_stroke_code) {
        case SD_SPHERE_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_sphere_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_SPHERE_SMOOTH_OP_SUBSTRACTION:{
            result_surface = eval_preview_stroke_sphere_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_SPHERE_SMOOTH_OP_PAINT:{
            result_surface = eval_preview_stroke_sphere_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_box_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_box_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_box_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CAPSULE_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_capsule_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CAPSULE_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_capsule_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CAPSULE_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_capsule_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CONE_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_cone_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CONE_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_cone_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CONE_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_cone_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CYLINDER_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_cylinder_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CYLINDER_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_cylinder_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_CYLINDER_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_cylinder_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_TORUS_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_torus_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_TORUS_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_torus_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_TORUS_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_torus_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_VESICA_SMOOTH_OP_UNION: {
            result_surface = eval_preview_stroke_vesica_union(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_VESICA_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_preview_stroke_vesica_substraction(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        case SD_VESICA_SMOOTH_OP_PAINT: {
            result_surface = eval_preview_stroke_vesica_paint(position, result_surface, edit_starting_index, edit_count);
            break;
        }
        default: {}
    }

    return result_surface;
}