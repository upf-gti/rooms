#include sdf_interval_functions.wgsl

/*
 _____       _                   
/  ___|     | |                  
\ `--. _ __ | |__   ___ _ __ ___ 
 `--. \ '_ \| '_ \ / _ \ '__/ _ \
/\__/ / |_) | | | |  __/ | |  __/
\____/| .__/|_| |_|\___|_|  \___|
      | |                        
      |_|
*/

fn eval_interval_stroke_sphere_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = cut_sphere_interval(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation);
            result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = sphere_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
            result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
        }
    }

    return result_surface;
}

fn eval_interval_stroke_sphere_smooth_substract(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = cut_sphere_interval(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation);
            result_surface = opSmoothSubtractionInterval(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = sphere_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
            result_surface = opSmoothSubtractionInterval(result_surface, tmp_surface, smooth_factor);
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

fn eval_interval_stroke_box_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    for(var i : u32 = starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = edit_list[i];
        tmp_surface = box_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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

fn eval_interval_stroke_capsule_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    for(var i : u32 = starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = edit_list[i];
        tmp_surface = capsule_interval(position, curr_edit.position, curr_edit.dimensions.x, curr_edit.dimensions.y, curr_edit.rotation);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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

fn eval_interval_stroke_cone_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    for(var i : u32 = starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = edit_list[i];
        tmp_surface = cone_interval(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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

fn eval_interval_stroke_cylinder_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    for(var i : u32 = starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = edit_list[i];
        tmp_surface = cylinder_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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

fn eval_interval_stroke_torus_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;
    
    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    if(cap_value > 0.0) {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = capped_torus_interval(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation);
            result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = starting_idx; i < ending_idx; i++) {
            let curr_edit : Edit = edit_list[i];
            tmp_surface = torus_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
            result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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


fn eval_interval_stroke_vesica_smooth_union(position : mat3x3f, current_surface : vec2f, stroke_idx : u32,  dimension_margin : vec4f) -> vec2f
{
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let curr_stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let edit_count : u32 = curr_stroke.edit_count;
    let parameters : vec4f = curr_stroke.parameters;
    let smooth_factor : f32 = parameters.w;
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = curr_stroke.edit_list_index;
    let ending_idx : u32 = curr_stroke.edit_count + starting_idx;

    for(var i : u32 = starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = edit_list[i];
        tmp_surface = vesica_interval(position, curr_edit.position, curr_edit.dimensions, curr_edit.rotation);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
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

fn evaluate_stroke_interval( position: mat3x3f, stroke_idx: u32, current_surface : vec2f, center : vec3f, half_size : f32) -> vec2f
{
    let stroke : ptr<storage, Stroke> = &stroke_history.strokes[stroke_idx];
    let stroke_operation : u32 = (*stroke).operation;
    let stroke_primitive : u32 = (*stroke).primitive;

    let curr_stroke_code : u32 = stroke_primitive | (stroke_operation << 4u);

    var result_surface : vec2f = current_surface;
    let initial_surface : vec2f = vec2f(10000.0, 10000.0);

    let edit_count : u32 = (*stroke).edit_count;
    let parameters : vec4f = (*stroke).parameters;

    let smooth_factor : f32 = parameters.w;

    switch(curr_stroke_code) {
        case SD_SPHERE_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_sphere_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_SPHERE_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_sphere_smooth_substract(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_BOX_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_box_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_BOX_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_box_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        case SD_CAPSULE_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_capsule_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_CAPSULE_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_capsule_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        case SD_CONE_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_cone_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_CONE_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_cone_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        case SD_CYLINDER_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_cylinder_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_CYLINDER_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_cylinder_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        case SD_TORUS_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_torus_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_TORUS_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_torus_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        case SD_VESICA_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_vesica_smooth_union(position, current_surface, stroke_idx, vec4f(0.0));
            break;
        }
        case SD_VESICA_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_vesica_smooth_union(position, initial_surface, stroke_idx, vec4f(0.0));
            result_surface = opSmoothSubtractionInterval(current_surface, result_surface, smooth_factor);
            break;
        }
        default: {}
    }

    return result_surface;
}