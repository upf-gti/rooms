
// Interval operations

fn iavec3_vecs(x : vec2f, y : vec2f, z : vec2f) -> mat3x3f
{
	return mat3x3f(vec3f(x, 0.0), vec3f(y, 0.0), vec3f(z, 0.0));
}

fn iavec3_vec(p : vec3f) -> mat3x3f
{
	return mat3x3f(vec3f(p.xx, 0.0), vec3f(p.yy, 0.0), vec3f(p.zz, 0.0));
}

fn imin(a : vec2f, b : vec2f) -> vec2f
{
	return vec2f(min(a.x,b.x),min(a.y,b.y));
}

fn imax(a : vec2f, b : vec2f) -> vec2f
{
	return vec2f(max(a.x,b.x),max(a.y,b.y));
}

fn imax_mats(v : mat3x3f, q : mat3x3f) -> mat3x3f
{ 
    return iavec3_vecs(
		imax(v[0].xy, q[0].xy),
		imax(v[1].xy, q[1].xy),
		imax(v[2].xy, q[2].xy));
}

fn iadd_vecs(a : vec2f, b : vec2f) -> vec2f
{
	return a + b;
}

fn iadd_mats(a : mat3x3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		a[0].xy + b[0].xy,
		a[1].xy + b[1].xy,
		a[2].xy + b[2].xy);
}

fn iadd_vec_mat(a : vec3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		a.xx + b[0].xy,
		a.yy + b[1].xy,
		a.zz + b[2].xy);
}

fn isub_vec_float(a : vec2f, b : f32) -> vec2f
{
	return a - b;
}

fn isub_vecs(a : vec2f, b : vec2f) -> vec2f
{
	return a - b.yx;
}

fn isub_mats(a : mat3x3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		a[0].xy - b[0].yx,
		a[1].xy - b[1].yx,
		a[2].xy - b[2].yx);
}

fn isub_mat_vec(a : mat3x3f, b : vec3f) -> mat3x3f
{
	return iavec3_vecs(
		a[0].xy - b.xx,
		a[1].xy - b.yy,
		a[2].xy - b.zz);
}

fn imul_vec_float(a : vec2f, c : f32) -> vec2f
{
	let b : vec2f = vec2f(c);
	let f : vec4f = vec4f(
		a.xxyy * b.xyxy
	);	
	return vec2f(
		min(min(f[0],f[1]),min(f[2],f[3])),
		max(max(f[0],f[1]),max(f[2],f[3])));
}

fn imul_vec2_vec2(a : vec2f, b : vec2f) -> vec2f
{
	let f : vec4f = vec4f(
		a.xxyy * b.xyxy
	);	
	return vec2f(
		min(min(f[0],f[1]),min(f[2],f[3])),
		max(max(f[0],f[1]),max(f[2],f[3])));
}

fn imul_float_vec(a : f32, b : vec2f) -> vec2f
{
	let f : vec2f = vec2f(a*b);	
	return vec2f(
		min(f[0],f[1]),
		max(f[0],f[1]));
}

fn ifix_interval(a : vec2f) -> vec2f
{
    return vec2f(
		min(a[0],a[1]),
		max(a[0],a[1])
    );
}

fn imul_mats(a : mat3x3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		imul_vec2_vec2(a[0].xy, b[0].xy),
		imul_vec2_vec2(a[1].xy, b[1].xy),
		imul_vec2_vec2(a[2].xy, b[2].xy)
	);
}

fn imul_float_mat(a : f32, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		imul_float_vec(a, b[0].xy),
		imul_float_vec(a, b[1].xy),
		imul_float_vec(a, b[2].xy)
	);
}

fn imul_vec2_mat(a : vec2f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		imul_vec2_vec2(a, b[0].xy),
		imul_vec2_vec2(a, b[1].xy),
		imul_vec2_vec2(a, b[2].xy)
	);
}

fn imul_vec3_mat(a : vec3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		imul_vec2_vec2(a.xx, b[0].xy),
		imul_vec2_vec2(a.yy, b[1].xy),
		imul_vec2_vec2(a.zz, b[2].xy)
	);
}

fn imul_vec3_vec2(a : vec3f, b : vec2f) -> mat3x3f
{
	return iavec3_vecs(
		imul_float_vec(a.x, b),
		imul_float_vec(a.y, b),
		imul_float_vec(a.z, b)
	);
}


fn idiv_vecs(a : vec2f, b : vec2f) -> vec2f
{
	let f : vec4f = vec4f(
		a.x/b, a.y/b
	);
	return vec2f(
		min(min(f[0],f[1]),min(f[2],f[3])),
		max(max(f[0],f[1]),max(f[2],f[3])));
}

fn idiv_mats(a : mat3x3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		idiv_vecs(a[0].xy, b[0].xy),
		idiv_vecs(a[1].xy, b[1].xy),
		idiv_vecs(a[2].xy, b[2].xy)
	);
}

fn isqrt(a : vec2f) -> vec2f
{
	return vec2f(sqrt(a.x),sqrt(a.y));
}

fn ipow2_vec(a : vec2f) -> vec2f
{	
    if (a.x >= 0.0) {
        return vec2f(a*a);
    } else
    if (a.y < 0.0) {
        return (a*a).yx;
    } else {
        return vec2f(0.0, max(a.x * a.x, a.y * a.y));
    }

	// return select(select(vec2f(0.0,max(a.x*a.x,a.y*a.y)), vec2f((a*a).yx), (a.y<0.0)), vec2f(a*a), (a.x>=0.0));
}

fn iinv(a : vec2f) -> vec2f {
    var inverted : vec2f;
    if (a.x > 0.0 || a.y < 0.0) {
        inverted = vec2f(1.0 / a.y, 1.0 / a.x);
    } else if (a.x < 0.0 && a.y > 0.0) {
        inverted =  vec2f(-10000.0, 10000.0);
    } else if (a.y == 0.0) {
        inverted =  vec2f(-10000.0, 1.0 / a.x);
    } else {
        inverted = vec2f(1.0 / a.y, 10000.0);
    }


    // Will never reach this
    return inverted;
}

// only valid for even numbers
fn ipow_vec(a : vec2f, n : f32) -> vec2f
{
    if (a.x >= 0.0) {
        return pow(a, vec2f(n));
    } else
    if (a.y < 0.0) {
        return pow(a, vec2f(n)).yx;
    } else {
        return vec2f(0.0, max(pow(a.x, n), pow(a.y, n)));
    }

	// return select(select(vec2f(0.0,max(a.x*a.x,a.y*a.y)), vec2f((a*a).yx), (a.y<0.0)), vec2f(a*a), (a.x>=0.0));
}

fn ipow2_mat(v : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
		ipow2_vec(v[0].xy),
		ipow2_vec(v[1].xy),
		ipow2_vec(v[2].xy));
}

fn ilensq(a : mat3x3f) -> vec2f
{
	let c : mat3x3f = ipow2_mat(a);
	return c[0].xy + c[1].xy + c[2].xy;
}

fn ilength(a : mat3x3f) -> vec2f
{
	let c : mat3x3f = ipow2_mat(a);
	return isqrt(c[0].xy + c[1].xy + c[2].xy);
}

fn icross_mats(a : mat3x3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
        isub_vecs(imul_vec2_vec2(a[1].xy, b[2].xy), imul_vec2_vec2(b[1].xy, a[2].xy)),
        isub_vecs(imul_vec2_vec2(a[2].xy, b[0].xy), imul_vec2_vec2(b[2].xy, a[0].xy)),
        isub_vecs(imul_vec2_vec2(a[0].xy, b[1].xy), imul_vec2_vec2(b[0].xy, a[1].xy))
    );
}

fn icross_vec_mat(a : vec3f, b : mat3x3f) -> mat3x3f
{
	return iavec3_vecs(
        isub_vecs(imul_float_vec(a.y, b[2].xy), imul_vec_float(b[1].xy, a.z)),
        isub_vecs(imul_float_vec(a.z, b[0].xy), imul_vec_float(b[2].xy, a.x)),
        isub_vecs(imul_float_vec(a.x, b[1].xy), imul_vec_float(b[0].xy, a.y))
    );
}

fn irotate_point_quat(position : mat3x3f, rotation : vec4f) -> mat3x3f
{
    let crosses : mat3x3f = icross_vec_mat(rotation.xyz, iadd_mats(icross_vec_mat(rotation.xyz, position), imul_float_mat(rotation.w, position)));
    let position_rotated : mat3x3f = iadd_mats(position, imul_float_mat(2.0, crosses));

    return position_rotated;
}

fn idot(a : mat3x3f, b : mat3x3f) -> vec2f
{
	let c : mat3x3f = imul_mats(a,b);
	return c[0].xy + c[1].xy + c[2].xy;
}

fn icontains(a : vec2f, v : f32) -> bool
{
	return ((v >= a.x) && (v < a.y));
}

fn ineg(a : vec2f) -> vec2f
{
	return vec2f(-a.y, -a.x);
}

fn ineg_mats(v : mat3x3f) -> mat3x3f
{ 
    return iavec3_vecs(
		ineg(v[0].xy),
		ineg(v[1].xy),
		ineg(v[2].xy));
}

fn iabs(a : vec2f) -> vec2f
{ 
    if (a.x >= 0.0) {
        return a;
    }
    
    if (a.y <= 0.0) {
        return vec2f(-a.y, -a.x);
    }
    
    return vec2f(0.0, max(-a.x, a.y));
}

fn iabs_mats(v : mat3x3f) -> mat3x3f
{ 
    return iavec3_vecs(
		iabs(v[0].xy),
		iabs(v[1].xy),
		iabs(v[2].xy));
}


fn ilessthan(a : vec2f, b : vec2f) -> vec2<bool> 
{    
    if (a.y < b.x) {
        return vec2<bool>(true, true);
    }
    
    if (a.x >= b.y) {
        return vec2<bool>(false, false);
    }
    
    return vec2<bool>(false, true);
}

fn iselect(a : vec2f, b : vec2f, cond : vec2<bool>) -> vec2f
{    
    if (cond.x) {
        return b;
    }
    
    if (!cond.y) {
        return a;
    }
    
    return vec2f(min(a.x, b.x), max(a.y, b.y));
}

fn irotate_interval_mats(p : mat3x3f, q : vec4f) -> mat3x3f {
    let v1 : vec3f = rotate_point_quat(vec3f(p[0].x, p[1].x, p[2].x), q);
    let v2 : vec3f = rotate_point_quat(vec3f(p[0].y, p[1].y, p[2].y), q);
    return iavec3_vecs(vec2f(v1.x, v2.x), vec2f(v1.y, v2.y), vec2f(v1.z, v2.z));
}

// Interval sdfs

fn sminN_interval( a : vec2f, b : vec2f, k : f32, n : f32 ) -> vec2f
{
    let h : vec2f = imul_float_vec(1.0 / k, imax(k + ineg(iabs(isub_vecs(a, b))), vec2f(0.0)));
    let m : vec2f = imul_float_vec(0.5, ipow_vec(h, n));
    let s : vec2f = imul_float_vec(k / n, m);

    return vec2f( iselect( isub_vecs(b, s), isub_vecs(a, s), ilessthan(a, b)));
}

fn isoft_min(a : vec2f, b : vec2f, r : f32) -> vec2f 
{ 
    let e : vec2f = imax(r + ineg(iabs(isub_vecs(a, b))), vec2f(0.0)); 
    return isub_vecs(imin(a, b), imul_float_vec(0.25 / r, imul_vec2_vec2(e, e))); 
}

fn isoft_min_poly(a : vec2f, b : vec2f, k : f32) -> vec2f {
    let h : vec2f = imul_float_vec(1.0 / k, imax(k + ineg(iabs(isub_vecs(a, b))), vec2f(0.0)));
    let m : vec2f = ipow2_vec(h);
    let s : vec2f = imul_float_vec(k * 0.25, m);

    return vec2f( iselect( isub_vecs(b, s), isub_vecs(a, s), ilessthan(a, b)));
}

fn opUnionInterval( s1 : vec2f, s2 : vec2f ) -> vec2f
{ 
    return imin( s1, s2 );
}
 
fn opSmoothUnionInterval( s1 : vec2f, s2 : vec2f, k : f32 ) -> vec2f
{
    return isoft_min(s2, s1, k);
}

fn opSmoothSubtractionInterval( s1 : vec2f, s2 : vec2f, k : f32 ) -> vec2f
{
    return ineg(isoft_min(s2, ineg(s1), k));
}

fn opSubtractionInterval( s1 : vec2f, s2 : vec2f ) -> vec2f
{
   return imax(s1, ineg(s2));
}

fn imat_add_to_upper(p : mat3x3f, v : vec3f) -> mat3x3f {
    var p_edit : mat3x3f = p;
    p_edit[0].x += v.x;
    p_edit[1].x += v.y;
    p_edit[2].x += v.z;

    return p_edit;
}

fn imat_add_to_lower(p : mat3x3f, v : vec3f) -> mat3x3f {
    var p_edit : mat3x3f = p;
    p_edit[0].y += v.x;
    p_edit[1].y += v.y;
    p_edit[2].y += v.z;

    return p_edit;
}

fn idot_mat(v1 : mat3x3f, v2 : mat3x3f) -> vec2f {
    return iadd_vecs(iadd_vecs(
		imul_vec2_vec2(v1[0].xy, v2[0].xy),
		imul_vec2_vec2(v1[1].xy, v2[1].xy)),
		imul_vec2_vec2(v1[2].xy, v2[2].xy));
}


fn sphere_interval(p : mat3x3f, offset : vec3f, r : f32) -> vec2f
{
	// x^2 + y^2 + z^2 - r^2
	return isub_vecs(ilensq(isub_mat_vec(p, offset)), vec2f(r*r));
}

// TODO rotation
fn box_interval(p : mat3x3f, edit_pos : vec3f, rotation : vec4f, size : vec3f) -> vec2f {
    let mat_zero_interval : mat3x3f = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));

    // Move edit
    let interval_translated : mat3x3f = irotate_point_quat(isub_mat_vec(p, edit_pos), rotation);

    let interval_translated_sized : mat3x3f = isub_mat_vec(iabs_mats(interval_translated), size);

    var i_distance : vec2f = ilength(imax_mats(interval_translated_sized, mat_zero_interval));
    var max_on_all_axis : vec2f = imax(interval_translated_sized[0].xy, imax(interval_translated_sized[1].xy, interval_translated_sized[2].xy));
    var i_dist2 : vec2f =  imin(max_on_all_axis , vec2f(0.0, 0.0));

    return iadd_vecs(i_distance, i_dist2);
}

fn isign_vec2(v : vec2f) -> vec2f {
    var s : vec2f = vec2f(1.0, 1.0);

    if (v.x < 0.0) {
        s.x = -1.0;
    }
    
    if (v.y < 0.0) {
        s.y = -1.0;
    }

    return s ; //vec2f(min(s.x, s.y), max(s.x, s.y));;
}

// sdCylinder(p : vec3f, a : vec3f, b : vec3f, rotation : vec4f, r : f32, rr : f32, color : vec3f) -> Surface
fn cylinder_interval(p : mat3x3f, start_pos : vec3f, rotation : vec4f, radius : f32, height : f32) -> vec2f {
    let cyl_origin : mat3x3f = isub_mat_vec(p, start_pos);
    
    let d_x : vec2f = isub_vec_float(iabs(isqrt(ipow2_vec(cyl_origin[0].xy) + ipow2_vec(cyl_origin[2].xy))), height); 
    let d_y : vec2f = isub_vec_float(iabs(cyl_origin[1].xy), height); 

    let max_d_x : vec2f = imax(vec2f(0.0), d_x);
    let max_d_y : vec2f = imax(vec2f(0.0), d_y);

    return iadd_vecs(imin(imax(d_x, d_y), vec2f(0.0)), isqrt(ipow2_vec(max_d_x.xy) + ipow2_vec(max_d_y.xy)));

    
    // let cyl_lenght : mat3x3f = iadd_vec_mat(cylinder_end_pos, ineg_mats(cyl_origin));

    // let cyl_lenght_square : vec2f = idot_mat(cyl_lenght, cyl_lenght);
    // let cyl_origin_facing_length : vec2f = idot_mat(cyl_origin, cyl_lenght);

    // var x : vec2f = ilength(isub_mats(imul_vec2_mat(cyl_lenght_square, cyl_origin), imul_vec2_mat(cyl_origin_facing_length, cyl_lenght)));
    // x = isub_vecs(x, imul_vec_float(cyl_lenght_square, radius));

    // let y : vec2f = isub_vecs(iabs(imul_float_vec(0.5, isub_vecs(cyl_lenght_square, cyl_origin_facing_length))), imul_float_vec(0.5, cyl_lenght_square));

    // let xx : vec2f = ipow2_vec(x);
    // let yy : vec2f = imul_vec2_vec2(ipow2_vec(y), cyl_lenght_square);

    // let interval_0 : vec2f = vec2f(0.0);

    // let d1 : vec2f = iadd_vecs(iselect(interval_0, xx, x > interval_0), iselect(interval_0, yy, y > interval_0));
    // let d2 : vec2f = ineg(imin(xx, yy));
    // let d : vec2f = iselect(d1, d2, imax(x, y) < interval_0);

    // AQUI
    //return idiv_vecs(imul_vec2_vec2(isign_vec2(d), isqrt(iabs(d))), cyl_lenght_square);

    //return idiv_vecs(imul_vec2_vec2(isign_vec2(d), isqrt(iabs(d))), cyl_lenght_square);
}

fn eval_edit_interval( p_x : vec2f, p_y : vec2f, p_z : vec2f,  primitive : u32, operation : u32, edit_parameters : vec4f, current_interval : vec2f, edit : Edit) -> vec2f
{
    var pSurface : vec2f;

    // Center in texture (position 0,0,0 is just in the middle)
    var size : vec3f = edit.dimensions.xyz;
    var radius : f32 = edit.dimensions.x;
    var size_param : f32 = edit.dimensions.w;
    var cap_value : f32 = edit_parameters.y;

    var onion_thickness : f32 = edit_parameters.x;
    let do_onion = onion_thickness > 0.0;

    let smooth_factor : f32 = edit_parameters.w;

    switch (primitive) {
        case SD_SPHERE: {
            onion_thickness = map_thickness( onion_thickness, radius );
            radius -= onion_thickness; // Compensate onion size
            // -1..1 no cap..fully capped
            if(cap_value > -1.0) { 
                // pSurface = sdCutSphere(position, edit.position, edit.rotation, radius, radius * cap_value * 0.999, edit.color);
            } else {
                pSurface = sphere_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, radius);
            }
            break;
        }
        case SD_BOX: {
            onion_thickness = map_thickness( onion_thickness, size.x );
            size_param = (size_param / 0.1) * size.x; // Make Rounding depend on the side length

            // Compensate onion size (Substract from box radius bc onion will add it later...)
            size -= onion_thickness;
            size_param -= onion_thickness; 

            pSurface = box_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, size);
            break;
        }
        case SD_CAPSULE: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            var height = radius; // ...
            // pSurface = sdCapsule(position, edit.position, edit.position - vec3f(0.0, 0.0, height), edit.rotation, size_param, edit.color);
            break;
        }
        case SD_CONE: {
            onion_thickness = map_thickness( onion_thickness, 0.01 );
            cap_value = cap_value * 0.5 + 0.5;
            radius = max(radius * (1.0 - cap_value), 0.0025);
            var dims = vec2f(size_param, size_param * cap_value);
            // pSurface = sdCone(position, edit.position,  edit.position - vec3f(0.0, 0.0, radius), edit.rotation, dims, edit.color);
            break;
        }
        // case SD_PYRAMID: {
        //     pSurface = sdPyramid(position, edit.position, edit.rotation, radius, size_param, edit.color);
        //     break;
        // }
        case SD_CYLINDER: {
            //onion_thickness = map_thickness( onion_thickness, size_param );
            //size_param -= onion_thickness; // Compensate onion size
            // pSurface = sdCylinder(position, edit.position,  edit.position - vec3f(0.0, 0.0, radius), edit.rotation, size_param, 0.0, edit.color);
            pSurface = cylinder_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, size.y, radius);
            break;
        }
        case SD_TORUS: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            size_param = clamp( size_param, 0.0001, radius );
            if(cap_value > -1.0) { // // -1..1 no cap..fully capped
                cap_value = cap_value * 0.5 + 0.5;
                var an = M_PI * (1.0 - cap_value);
                var angles = vec2f(sin(an), cos(an));
                // pSurface = sdCappedTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, angles, edit.color);
            } else {
                // pSurface = sdTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, edit.color);
            }
            break;
        }
        default: {
            break;
        }
    }

    // Shape edition ...
    if( do_onion && (operation == OP_UNION || operation == OP_SMOOTH_UNION) )
    {
        // pSurface = opOnion(pSurface, onion_thickness);
    }

    switch (operation) {
        case OP_UNION: {
            pSurface = opUnionInterval(current_interval, pSurface);
            break;
        }
        case OP_SUBSTRACTION:{
            pSurface = opSubtractionInterval(current_interval, pSurface);
            break;
        }
        case OP_INTERSECTION: {
            // pSurface = opIntersection(current_interval, pSurface);
            break;
        }
        case OP_PAINT: {
            // pSurface = opPaint(current_interval, pSurface, edit.color);
            break;
        }
        case OP_SMOOTH_UNION: {
            pSurface = opSmoothUnionInterval(current_interval, pSurface, smooth_factor);
            break;
        }
        case OP_SMOOTH_SUBSTRACTION: {
            pSurface = opSmoothSubtractionInterval(current_interval, pSurface, smooth_factor);
            break;
        }
        case OP_SMOOTH_INTERSECTION: {
            // pSurface = opSmoothIntersection(current_interval, pSurface, SMOOTH_FACTOR);
            break;
        }
        case OP_SMOOTH_PAINT: {
            // pSurface = opSmoothPaint(current_interval, pSurface, edit.color, SMOOTH_FACTOR);
            break;
        }
        default: {
            break;
        }
    }

    return pSurface;
}