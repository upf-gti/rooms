#include ../color_blend_modes.wgsl
#include sdf_commons.wgsl

// Data containers
struct Material {
    albedo      : vec3f,
    roughness   : f32,
    metalness   : f32
};

struct Surface {
    material    : Material,
    distance    : f32
};

// Material operation storages
fn Material_mult_by(m : Material, v : f32) -> Material {
    return Material(m.albedo * v, m.roughness * v, m.metalness * v);
}

fn Material_sum_Material(m1 : Material, m2 : Material) -> Material {
    return Material(m1.albedo + m2.albedo, m1.roughness + m2.roughness, m1.metalness + m2.metalness);
}

fn Material_mix(m1 : Material, m2 : Material, t : f32) -> Material {
    return Material_sum_Material(Material_mult_by(m1, 1.0 - t), Material_mult_by(m2, t));
}

// Other Primitives

fn sdPlane( p : vec3f, c : vec3f, n : vec3f, h : f32, material : Material) -> Surface
{
    // n must be normalized
    var sf : Surface;
    sf.distance = dot(p - c, n) + h;
    sf.material = material;
    return sf;
}

fn sdPyramid( p : vec3f, c : vec3f, rotation : vec4f, r : f32, h : f32, material : Material) -> Surface
{
    var sf : Surface;
    let m2 : f32 = h * h + 0.25;

    let pos : vec3f = rotate_point_quat(p - c, rotation);

    let abs_pos : vec2f = abs(pos.xz);
    let swizzle_pos : vec2f = select(abs_pos.xy, abs_pos.yx, abs_pos.y > abs_pos.x);
    let moved_pos : vec3f = vec3f(swizzle_pos.x - 0.5, pos.y, swizzle_pos.y - 0.5);

    let q : vec3f = vec3(moved_pos.z, h * moved_pos.y - 0.5 * moved_pos.x, h * moved_pos.x + 0.5 * moved_pos.y);

    let s : f32 = max(-q.x, 0.0);
    let t : f32 = clamp((q.y - 0.5 * moved_pos.z) / (m2 + 0.25), 0.0, 1.0);

    let a : f32 = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b : f32 = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

    let d2 : f32 = select(min(a, b), 0.0, min(q.y, -q.x * m2 - q.y * 0.5) > 0.0);

    sf.distance = sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -moved_pos.y)) - r;
    sf.material = material;
    return sf;
}

// Primitive combinations

fn colorMix( a : vec3f, b : vec3f, n : f32 ) -> vec3f
{
    let aa : vec3f = a * a;
    let bb : vec3f = b * b;
    return sqrt(mix(aa, bb, n));
}

fn sminN( a : f32, b : f32, k : f32, n : f32 ) -> vec2f
{
    let h : f32 = max(k - abs(a - b), 0.0) / k;
    let m : f32 = pow(h, n) * 0.5;
    let s : f32 = m * k / n;
    if (a < b) {
        return vec2f(a - s, m);
    } else {
        return vec2f(b - s, 1.0 - m);
    }
}

fn soft_min(a : f32, b : f32, k : f32) -> vec2f 
{ 
    let h : f32 = max(k - abs(a - b), 0) / k; 
    let m : f32 = h * h * 0.5;
    return vec2f(min(a, b) - h * h * k * 0.25, select(1.0 - m, m, a < b)); 
}

// From iqulzes and Dreams
fn sminQuadratic(a : f32, b : f32, k : f32) -> vec2f {
    let norm_k : f32 = max(k, 0.001);// * 4.0;
    let h : f32 = max(norm_k - abs(a - b), 0.0) / norm_k;
    let m : f32 = h*h;
    let s : f32 = m*norm_k * 0.25;

    if (a < b) {
        return vec2f(a - s, m *0.5);
    } else {
        return vec2f(b - s, 1.0 - (m * 0.5));
    }
}

fn opSmoothUnion( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    //let smin : vec2f = soft_min(s2.distance, s1.distance, k);
    let smin : vec2f = sminQuadratic(s2.distance, s1.distance, k);
    var sf : Surface;
    sf.distance = smin.x;
    sf.material = Material_mix(s2.material, s1.material, smin.y);
    return sf;
}

fn minSurface( s1 : Surface, s2 : Surface ) -> Surface
{ 
    if ( s1.distance < s2.distance ) {
        return s1;
    } else {
        return s2;
    } 
}

fn maxSurface( s1 : Surface, s2 : Surface ) -> Surface
{ 
    if ( s1.distance > s2.distance ) {
        return s1;
    } else {
        return s2;
    } 
}

fn opUnion( s1 : Surface, s2 : Surface ) -> Surface
{ 
    return minSurface( s1, s2 );
}

fn opSubtraction( s1 : Surface, s2 : Surface ) -> Surface
{ 
    var s2neg : Surface = s2;
    s2neg.distance = -s2neg.distance;
    var s : Surface = maxSurface( s1, s2neg );
    //s.color = s1.color;
    return s;
}

fn opIntersection( s1 : Surface, s2 : Surface ) -> Surface
{ 
    var s : Surface = maxSurface( s1, s2 );
    s.material = s1.material;
    return s;
}

fn opPaint( s1 : Surface, s2 : Surface, material : Material, color_blend_op : u32 ) -> Surface
{
    var sColorInter : Surface = opIntersection(s1, s2);
    var new_mat : Material;

    var base_color : vec3f = s1.material.albedo;
    var new_layer_color : vec3f = material.albedo;
    var result_color : vec3f = new_layer_color;

    if(color_blend_op == CBM_MULTIPLY) {
        result_color = multiply(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_SCREEN) {
        result_color = screen(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_DARKEN) {
        result_color = darken(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_LIGHTEN) {
        result_color = lighten(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_ADDITIVE) {
        result_color = additive(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_MIX) {
        result_color = mix(base_color, new_layer_color, 0.5);
    }

    if(color_blend_op > 0u) {
        // Since we add too many edits in smear, we need to force blending colors
        // to be more realistic..
        result_color = mix(base_color, result_color, 0.5);
    }

    new_mat.albedo = clamp(result_color, vec3f(0.0), vec3f(1.0));
    new_mat.roughness = material.roughness;
    new_mat.metalness = material.metalness;

    sColorInter.material = new_mat;
    return opUnion(s1, sColorInter);
}

fn opSmoothSubtraction( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let smin : vec2f = sminQuadratic(s2.distance, -s1.distance, k);
    var s : Surface;
    s.distance = -smin.x;
    s.material = Material_mix(s2.material, s1.material, smin.y);
    return s;
}

fn opSmoothIntersection( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let h : f32 = max(k - abs(s1.distance - s2.distance), 0.0);
    var s : Surface;
    s.distance = max(s1.distance, s2.distance) + h * h * 0.25 / k;
    s.material = Material_mix(s1.material, s2.material, h);
    return s;
}

fn opSmoothPaint( s1 : Surface, s2 : Surface, color_blend_op : u32, material : Material, k : f32 ) -> Surface
{
    // Permorm the color blending on the incomming material, and then blend it with the other material
    var tmp_material : Material = material;

    let base_color : vec3f = s1.material.albedo;
    let new_layer_color : vec3f = material.albedo;

    if(color_blend_op == CBM_MIX) {
        tmp_material.albedo = mix(base_color, new_layer_color, 0.5);
    }
    else if(color_blend_op == CBM_ADDITIVE) {
        tmp_material.albedo = additive(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_MULTIPLY) {
        tmp_material.albedo = multiply(base_color, new_layer_color);
    }
    else if(color_blend_op == CBM_SCREEN) {
        tmp_material.albedo = screen(base_color, new_layer_color);
    }
    // else if(color_blend_op == CBM_DARKEN) {
    //     tmp_material.albedo = darken(base_color, new_layer_color);
    // }
    // else if(color_blend_op == CBM_LIGHTEN) {
    //     tmp_material.albedo = lighten(base_color, new_layer_color);
    // }

    if(color_blend_op > 0u) {
        // Since we add too many edits in smear, we need to force blending colors
        // to be more realistic..
        tmp_material.albedo = mix(base_color, tmp_material.albedo, 0.5);
    }

    tmp_material.albedo = clamp(tmp_material.albedo, vec3f(0.0), vec3f(1.0));

    let paint_smooth_factor : f32 = 0.005;
    let s_intersection : Surface = opSmoothIntersection(s1, s2, paint_smooth_factor);
    let smin : vec2f = sminQuadratic(s_intersection.distance, s1.distance, paint_smooth_factor);

    var s : Surface;
    s.distance = s1.distance;
    s.material = Material_mix(tmp_material, s1.material, pow(smin.y, 3.0));
    return s;
}

fn opOnion( s1 : Surface, t : f32 ) -> Surface
{
    var s : Surface;
    s.distance = abs(s1.distance) - t;
    s.material = s1.material;
    return s;
}

fn map_thickness( t : f32, v_max : f32 ) -> f32
{
    return select( 0.0, max(t * v_max * 0.375, 0.003), t > 0.0);
}

// TODO(Juan): Onion
// TODO(Juan): Unify materials

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

fn sdSphere( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material ) -> Surface
{
    let r : f32 = dims.x;
    var sf : Surface;
    sf.distance = length(p - c) - r;
    sf.material = material;
    return sf;
}

fn sdCutSphere( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material ) -> Surface
{
    var cap_value : f32 = clamp(parameters.y * 0.75, 0.0, 1.0) * 2.0 - 1.0;
    let r : f32 = dims.x;
    let h : f32 = r * cap_value;
    var sf : Surface;

    // sampling independent computations (only depend on shape)
    var w = sqrt(r * r - h * h);

    let pos : vec3f = rotate_point_quat(p - c, rotation);

    // sampling dependant computations
    var q = vec2f( length(pos.xy), pos.z );
    var s = max( (h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y );
    if(s < 0.0) {
        sf.distance = length(q) - r;
    } else if(q.x<w) {
        sf.distance = h - q.y;
    } else  {
        sf.distance = length(q - vec2f(w, h));
    }
                    
    sf.material = material;
    return sf;
}

fn eval_stroke_sphere_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;

    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;

    let smooth_factor : f32 = parameters.w;
    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }
    
    return result_surface;
}

fn eval_stroke_sphere_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32 ) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;

    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;

    let smooth_factor : f32 = parameters.w;
    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    /*if(cap_value > 0.0) {
        for(var i : u32 = 0u; i < edit_count; i++) {
            let curr_edit : Edit = edit_array[i];
            tmp_surface = sdCutSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }
    } else {
        for(var i : u32 = 0u; i < edit_count; i++) {
            let curr_edit : Edit = edit_array[i];
            tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
            result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
        }*/
    
    return result_surface;
}

fn eval_stroke_sphere_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32 ) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;

    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);
    let smooth_factor : f32 = parameters.w;
    let cap_value : f32 = parameters.y;

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdSphere(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
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

fn sdBox( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material ) -> Surface
{
    var sf : Surface;
    var size : vec3f = dims.xyz;
    
    let round : f32 =  clamp(dims.w / 0.08, 0.0, 1.0) * min(size.x, min(size.y, size.z));
    size -= round;

    let pos : vec3f = rotate_point_quat(p - c, rotation);
    let q : vec3f = abs(pos) - size;
    sf.distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - round;
    sf.material = material;
    return sf;
}

fn eval_stroke_box_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32 ) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;

    let smooth_factor : f32 = parameters.w;
    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_stroke_box_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32 ) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;

    let smooth_factor : f32 = parameters.w;
    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_stroke_box_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>, edit_starting_idx : u32, edit_count : u32 ) -> Surface {
    var result_surface : Surface = current_surface;
    var tmp_surface : Surface;
    
    let stroke_material = curr_stroke.material;
    let parameters : vec4f = curr_stroke.parameters;
    let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

    let smooth_factor : f32 = parameters.w;
    let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

    let starting_idx : u32 = edit_starting_idx;
    let ending_idx : u32 = edit_count + edit_starting_idx;

    for(var i : u32 = edit_starting_idx; i < ending_idx; i++) {
        let curr_edit : Edit = curr_edit_list[i];
        tmp_surface = sdBox(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
        result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
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

// fn sdCapsule( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
// {
//     var sf : Surface;
//     let r : f32 = dims.x;
//     let height : f32 = dims.y;
    
//     var pc : vec3f = rotate_point_quat(p - c, rotation);
//     pc.y -= clamp(pc.y, 0.0, height);

//     sf.distance = length(pc) - r;
//     sf.material = material;
//     return sf;
// }

// // #template_function eval_stroke_capsule_union
// // #TMP_SURFACE tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.rotation, size_param, height, material);
// // #RESULT_SURFACE result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
// // #end_template

// fn eval_stroke_capsule_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material  = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
    
//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_capsule_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_capsule_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
//     let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCapsule(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//     }

//     return result_surface;
// }

// /*
//  _____                  
// /  __ \                 
// | /  \/ ___  _ __   ___ 
// | |    / _ \| '_ \ / _ \
// | \__/\ (_) | | | |  __/
//  \____/\___/|_| |_|\___|
// */

// fn sdCone( p : vec3f, a : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
// {
//     var sf : Surface;

//     let cap_value : f32 = max(parameters.y, 0.0001);
//     var radius = dims.x;
//     var height : f32 = max(dims.y * (1.0 - cap_value), 0.0025);

//     let round : f32 = clamp(dims.w / 0.08, 0.0001, 1.0) * min(radius, height) * 0.5;
//     radius -= round;
//     height -= round;

//     let r1 = radius; // base radius
//     let r2 = radius * cap_value; // top radius
//     let h : f32 = height * 0.5;

//     let pos : vec3f = rotate_point_quat(p - a, rotation) - vec3f(0.0, h, 0.0);
//     let q = vec2f( length(pos.xz), pos.y );
//     let k1 = vec2f(r2, h);
//     let k2 = vec2f(r2-r1, 2.0 * h);
//     let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y<0.0)), abs(q.y) - h);
//     let cb = q - k1 + k2 * clamp( dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
//     let s : f32 = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);

//     sf.distance = s * sqrt(min(dot(ca, ca), dot(cb, cb))) - round;
//     sf.material = material;
//     return sf;
// }

// fn eval_stroke_cone_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material  = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
    
//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_cone_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_cone_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
//     let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCone(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//     }

//     return result_surface;
// }


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

// fn sdCylinder(p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
// {
//     var sf : Surface;
//     var r : f32 = dims.x;
//     var h : f32 = dims.y;

//     let round : f32 = clamp(dims.w / 0.08, 0.0001, 1.0) * min(r, h);
//     r -= round;
//     h -= round;

//     let pos : vec3f = rotate_point_quat(p - c, rotation);

//     let d : vec2f = abs(vec2f(length(vec2f(pos.x, pos.z)), pos.y)) - vec2(r, h);
//     sf.distance = min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - round;
//     sf.material = material;
//     return sf;
// }

// fn eval_stroke_cylinder_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material  = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
    
//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_cylinder_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_cylinder_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
//     let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdCylinder(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//     }

//     return result_surface;
// }

// /*
//  _____                    
// |_   _|                   
//   | | ___  _ __ _   _ ___ 
//   | |/ _ \| '__| | | / __|
//   | | (_) | |  | |_| \__ \
//   \_/\___/|_|   \__,_|___/
// */

// fn sdTorus( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
// {
//     let radius = dims.x;
//     let thickness = clamp( dims.y, 0.0001, radius );

//     var sf : Surface;
//     let pos : vec3f = rotate_point_quat(p - c, rotation);
//     var q = vec2f(length(pos.xz) - radius, pos.y);
//     sf.distance = length(q) - thickness;
//     sf.material = material;
//     return sf;
// }

// fn sdCappedTorus( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
// {
//     let cap_value : f32 = clamp(parameters.y, 0.0001, 0.999);
//     let theta : f32 = M_PI * (1.0 - cap_value);
//     let angles : vec2f = vec2f(sin(theta), cos(theta));
//     let radius : f32  = dims.x;
//     let thickness : f32 = clamp( dims.y, 0.0001, radius );

//     var sf : Surface;
//     var pos : vec3f = rotate_point_quat(p - c, rotation);
//     pos.x = abs(pos.x);

//     var k : f32 = select(length(pos.xz), dot(pos.xz, angles), angles.y * pos.x > angles.x * pos.z);

//     sf.distance = sqrt( dot(pos, pos) + radius * radius - 2.0 * radius * k ) - thickness;
//     sf.material = material;
//     return sf;
// }

// fn eval_stroke_torus_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material  = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
    
//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     let cap_value : f32 = parameters.y;

//     if(cap_value > 0.0) {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//         }
//     } else {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//         }
//     }

//     return result_surface;
// }

// fn eval_stroke_torus_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     let cap_value : f32 = parameters.y;

//     if(cap_value > 0.0) {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//         }
//     } else {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//         }
//     }

//     return result_surface;
// }

// fn eval_stroke_torus_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>, curr_edit_list : ptr<storage, array<Edit>>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &(curr_stroke.edits);
//     let edit_count : u32 = curr_stroke.edit_count;
//     let stroke_material = curr_stroke.material;
//     let parameters : vec4f = curr_stroke.parameters;
//     let stroke_blend_mode : u32 = curr_stroke.color_blend_op;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     let cap_value : f32 = parameters.y;

//     if(cap_value > 0.0) {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdCappedTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//         }
//     } else {
//         for(var i : u32 = 0u; i < edit_count; i++) {
//             let curr_edit : Edit = edit_array[i];
//             tmp_surface = sdTorus(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//             result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//         }
//     }

//     return result_surface;
// }

/*
 _   _           _           
| | | |         (_)          
| | | | ___  ___ _  ___ __ _ 
| | | |/ _ \/ __| |/ __/ _` |
\ \_/ /  __/\__ \ | (_| (_| |
 \___/ \___||___/_|\___\__,_|
*/

fn sdVesica( p : vec3f, c : vec3f, dims : vec4f, parameters : vec2f, rotation : vec4f, material : Material) -> Surface
{
    var sf : Surface;

    var radius : f32 = dims.x * 2.0;
    var height : f32 = dims.y * 2.0;

    let round : f32 = clamp(dims.w / 0.125, 0.001, 0.99) * min(radius, height);

    let pos : vec3f = rotate_point_quat(p - c, rotation) + vec3f(0.0, height * 0.5, 0.0);

    // shape constants
    let h : f32 = height * 0.5;
    let w : f32 = radius * 0.5;
    let d : f32 = 0.5 * (h * h - w * w) / w;
    
    // project to 2D
    let q : vec2f = vec2f(length(pos.xz), abs(pos.y - h));
    
    // feature selection (vertex or body)
    let t : vec3f = select(vec3f(-d,0.0,d+w), vec3f(0.0,h,0.0), (h*q.x < d*(q.y-h)));
    
    sf.distance = (length(q-t.xy) - t.z) - round;
    sf.material = material;
    return sf;
}

// fn eval_stroke_vesica_union( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
//     let edit_count : u32 = (*curr_stroke).edit_count;
//     let stroke_material  = (*curr_stroke).material;
//     let parameters : vec4f = (*curr_stroke).parameters;
    
//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothUnion(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_vesica_substraction( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
//     let edit_count : u32 = (*curr_stroke).edit_count;
//     let stroke_material = (*curr_stroke).material;
//     let parameters : vec4f = (*curr_stroke).parameters;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothSubtraction(result_surface, tmp_surface, smooth_factor);
//     }

//     return result_surface;
// }

// fn eval_stroke_vesica_paint( position : vec3f, current_surface : Surface, curr_stroke: ptr<storage, Stroke>) -> Surface {
//     var result_surface : Surface = current_surface;
//     var tmp_surface : Surface;

//     let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
//     let edit_count : u32 = (*curr_stroke).edit_count;
//     let stroke_material = (*curr_stroke).material;
//     let parameters : vec4f = (*curr_stroke).parameters;
//     let stroke_blend_mode : u32 = (*curr_stroke).color_blend_op;

//     let smooth_factor : f32 = parameters.w;
//     let material : Material = Material(stroke_material.color.xyz, stroke_material.roughness, stroke_material.metallic);

//     for(var i : u32 = 0u; i < edit_count; i++) {
//         let curr_edit : Edit = edit_array[i];
//         tmp_surface = sdVesica(position, curr_edit.position, curr_edit.dimensions, parameters.xy, curr_edit.rotation, material);
//         result_surface = opSmoothPaint(result_surface, tmp_surface, stroke_blend_mode, material, smooth_factor);
//     }

//     return result_surface;
// }

/*
______          _           
| ___ \        (_)          
| |_/ / ___ _____  ___ _ __ 
| ___ \/ _ \_  / |/ _ \ '__|
| |_/ /  __// /| |  __/ |   
\____/ \___/___|_|\___|_|
*/


// // IQ adaptation to 3d of http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
// // { dist, t, y (above the plane of the curve, x (away from curve in the plane of the curve))
// fn sdQuadraticBezier(p : vec3f, start : vec3f, cp : vec3f, end : vec3f, thickness : f32, rotation : vec4f, material : Material) -> Surface
// {
//     var b0 : vec3f = start - p;
//     var b1 : vec3f = cp - p;
//     var b2 : vec3f = end - p;
    
//     var b01 : vec3f = cross(b0, b1);
//     var b12 : vec3f = cross(b1, b2);
//     var b20 : vec3f = cross(b2, b0);
    
//     var n : vec3f =  b01 + b12 + b20;
    
//     var a : f32 = -dot(b20, n);
//     var b : f32 = -dot(b01, n);
//     var d : f32 = -dot(b12, n);

//     var m : f32 = -dot(n,n);
    
//     var g : vec3f =  (d-b)*b1 + (b+a*0.5)*b2 + (-d-a*0.5)*b0;
//     var f : f32 = a*a*0.25-b*d;
//     var k : vec3f = b0-2.0*b1+b2;
//     var t : f32 = clamp((a*0.5+b-0.5*f*dot(g,k)/dot(g,g))/m, 0.0, 1.0 );
    
//     var sf : Surface;
//     sf.distance = length(mix(mix(b0,b1,t), mix(b1,b2,t),t)) - thickness;
//     sf.material = material;
//     return sf;
// }

// STROKE EVALUATION ================

fn evaluate_stroke( position: vec3f, stroke: ptr<storage, Stroke, read>, curr_edit_list : ptr<storage, array<Edit>, read>, current_surface : Surface, edit_starting_index : u32, edit_count : u32) -> Surface {
    let stroke_operation : u32 = (*stroke).operation;
    let stroke_primitive : u32 = (*stroke).primitive;

    let curr_stroke_code : u32 = stroke_primitive | (stroke_operation << 4u);

    var result_surface : Surface = current_surface;

    switch(curr_stroke_code) {
        case SD_SPHERE_SMOOTH_OP_UNION: {
            result_surface = eval_stroke_sphere_union(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        case SD_SPHERE_SMOOTH_OP_SUBSTRACTION:{
            result_surface = eval_stroke_sphere_substraction(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        case SD_SPHERE_SMOOTH_OP_PAINT:{
            result_surface = eval_stroke_sphere_paint(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_UNION: {
            result_surface = eval_stroke_box_union(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_stroke_box_substraction(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        case SD_BOX_SMOOTH_OP_PAINT: {
            result_surface = eval_stroke_box_paint(position, result_surface, stroke, curr_edit_list, edit_starting_index, edit_count);
            break;
        }
        // case SD_CONE_SMOOTH_OP_UNION: {
        //     result_surface = eval_stroke_cone_union(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CONE_SMOOTH_OP_SUBSTRACTION: {
        //     result_surface = eval_stroke_cone_substraction(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CONE_SMOOTH_OP_PAINT: {
        //     result_surface = eval_stroke_cone_paint(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CAPSULE_SMOOTH_OP_UNION: {
        //     result_surface = eval_stroke_capsule_union(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CAPSULE_SMOOTH_OP_SUBSTRACTION: {
        //     result_surface = eval_stroke_capsule_substraction(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CAPSULE_SMOOTH_OP_PAINT: {
        //     result_surface = eval_stroke_capsule_paint(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CYLINDER_SMOOTH_OP_UNION: {
        //     result_surface = eval_stroke_cylinder_union(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CYLINDER_SMOOTH_OP_SUBSTRACTION: {
        //     result_surface = eval_stroke_cylinder_substraction(position, result_surface, stroke);
        //     break;
        // }
        // case SD_CYLINDER_SMOOTH_OP_PAINT: {
        //     result_surface = eval_stroke_cylinder_paint(position, result_surface, stroke);
        //     break;
        // }
        // case SD_TORUS_SMOOTH_OP_UNION: {
        //     result_surface = eval_stroke_torus_union(position, result_surface, stroke);
        //     break;
        // }
        // case SD_TORUS_SMOOTH_OP_SUBSTRACTION: {
        //     result_surface = eval_stroke_torus_substraction(position, result_surface, stroke);
        //     break;
        // }
        // case SD_TORUS_SMOOTH_OP_PAINT: {
        //     result_surface = eval_stroke_torus_paint(position, result_surface, stroke);
        //     break;
        // }
        // case SD_VESICA_SMOOTH_OP_UNION: {
        //     result_surface = eval_stroke_vesica_union(position, result_surface, stroke);
        //     break;
        // }
        // case SD_VESICA_SMOOTH_OP_SUBSTRACTION: {
        //     result_surface = eval_stroke_vesica_substraction(position, result_surface, stroke);
        //     break;
        // }
        // case SD_VESICA_SMOOTH_OP_PAINT: {
        //     result_surface = eval_stroke_vesica_paint(position, result_surface, stroke);
        //     break;
        // }
        default: {}
    }

    return result_surface;
}

fn evaluate_single_edit( position : vec3f, primitive : u32, operation : u32, parameters : vec4f, color_blend_op : u32, current_surface : Surface, stroke_material : Material, edit : Edit) -> Surface
{
    var pSurface : Surface;

    // Center in texture (position 0,0,0 is just in the middle)
    var size : vec3f = edit.dimensions.xyz;
    var radius : f32 = edit.dimensions.x;
    var size_param : f32 = edit.dimensions.w;

    // 0 no cap ... 1 fully capped
    var cap_value : f32 = clamp(parameters.y, 0.0, 0.99);

    var onion_thickness : f32 = parameters.x;
    let do_onion = onion_thickness > 0.0;

    let edit_parameters : vec2f = vec2f(onion_thickness, cap_value);

    let smooth_factor : f32 = parameters.w;

    switch (primitive) {
        case SD_SPHERE: {
            // onion_thickness = map_thickness( onion_thickness, radius );
            //radius -= onion_thickness; // Compensate onion size
            if(cap_value > 0.0) {
                pSurface = sdCutSphere(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
            } else {
                pSurface = sdSphere(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
            }
            break;
        }
        case SD_BOX: {
            // onion_thickness = map_thickness( onion_thickness, size.x );
            // size -= onion_thickness;
            // size_param -= onion_thickness;
            pSurface = sdBox(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
            break;
        }
        // case SD_CAPSULE: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     pSurface = sdCapsule(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     break;
        // }
        // case SD_CYLINDER: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     pSurface = sdCylinder(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     break;
        // }
        // case SD_TORUS: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     if(cap_value > 0.0) {
        //         pSurface = sdCappedTorus(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     } else {
        //         pSurface = sdTorus(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     }
        //     break;
        // }
        // case SD_VESICA: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     pSurface = sdVesica(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     break;
        // }
        // case SD_BEZIER: {
        //     var curve_thickness : f32 = 0.01;
        //     pSurface = sdQuadraticBezier(position, edit.position, edit.position + vec3f(0.1, 0.2, 0.0), edit.position + vec3f(0.2, 0.0, 0.0), curve_thickness, edit.rotation, stroke_material);
        //     break;
        // }
        // // case SD_PYRAMID: {
        // //     pSurface = sdPyramid(position, edit.position, edit.rotation, radius, size_param, edit_color);
        // //     break;
        // // }
        // case SD_CYLINDER: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     pSurface = sdCylinder(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     break;
        // }
        // case SD_TORUS: {
        //     // onion_thickness = map_thickness( onion_thickness, size_param );
        //     // size_param -= onion_thickness; // Compensate onion size
        //     if(cap_value > 0.0) {
        //         pSurface = sdCappedTorus(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     } else {
        //         pSurface = sdTorus(position, edit.position, edit.dimensions, edit_parameters, edit.rotation, stroke_material);
        //     }
        //     break;
        // }
        // // case SD_BEZIER: {
        // //     var curve_thickness : f32 = 0.01;
        // //     pSurface = sdQuadraticBezier(position, edit.position, edit.position + vec3f(0.1, 0.2, 0.0), edit.position + vec3f(0.2, 0.0, 0.0), curve_thickness, edit.rotation, stroke_material);
        // //     break;
        // // }
        default: {
            break;
        }
    }

    // // Shape edition ...
    // if( do_onion && (operation == OP_UNION || operation == OP_SMOOTH_UNION) )
    // {
    //     pSurface = opOnion(pSurface, onion_thickness);
    // }

    pSurface.material = stroke_material;

    switch (operation) {
        case OP_SMOOTH_UNION: {
            pSurface = opSmoothUnion(current_surface, pSurface, smooth_factor);
            break;
        }
        case OP_SMOOTH_SUBSTRACTION: {
            pSurface = opSmoothSubtraction(current_surface, pSurface, smooth_factor);
            break;
        }
        case OP_SMOOTH_INTERSECTION: {
            pSurface = opSmoothIntersection(current_surface, pSurface, smooth_factor);
            break;
        }
        case OP_SMOOTH_PAINT: {
            pSurface = opSmoothPaint(current_surface, pSurface, color_blend_op, stroke_material, smooth_factor);
            break;
        }
        default: {
            break;
        }
    }
    
    return pSurface;
}
