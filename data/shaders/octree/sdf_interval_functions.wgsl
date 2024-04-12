
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

fn imix(x : vec2f, y : vec2f, a: vec2f) -> vec2f
{
    return iadd_vecs(x, imul_vec2_vec2(isub_vecs(y, x), a));
}

fn imix_mats(x : mat3x3f, y : mat3x3f, a: vec2f) -> mat3x3f
{
    return iavec3_vecs(
		imix(x[0].xy, y[0].xy, a),
		imix(x[1].xy, y[1].xy, a),
		imix(x[2].xy, y[2].xy, a));
}

fn iclamp(a : vec2f, min : f32, max: f32) -> vec2f
{
	return imax(imin(a, vec2f(max)), vec2f(min));
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

fn iadd_mat_float(a : mat3x3f, b : f32) -> mat3x3f {
    return iavec3_vecs(
		a[0].xy + b,
		a[1].xy + b,
		a[2].xy + b);
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
}

fn ipow_n_vec(a : vec2f, n : f32) -> vec2f
{	
    if (a.x >= 0.0) {
        return vec2f(pow(a, vec2f(n)));
    } else
    if (a.y < 0.0) {
        return (pow(a, vec2f(n))).yx;
    } else {
        return vec2f(0.0, max(pow(a.x, n), pow(a.y, n)));
    }
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

fn igreaterthan(a : vec2f, b : vec2f) -> vec2<bool> 
{
    if (a.x > b.y) {
        return vec2<bool>(true, true);
    }

    if (a.y <= b.x) {
        return vec2<bool>(false, false);
    }

    return vec2<bool>(false, true);
}

fn ibool_minmax(x : bool, y : bool, z : bool, w : bool) -> vec2<bool>
{
    return vec2<bool>(
        x || y || z || w,
        x && y && z && w
    );
}

fn iand(x : vec2<bool>, y : vec2<bool>) -> vec2<bool>
{
    return ibool_minmax(x.x && y.x, x.x && y.y, x.y && y.x, x.y && y.y);
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

fn isoft_min_quadratic(a : vec2f, b : vec2f, k : f32) -> vec2f {
    let norm_k : f32 = k * 4.0;
    //  h = (1.0/k) * (max(k -abs(a-b), 0.0))
    let h : vec2f = imul_float_vec(1.0 / norm_k, imax(norm_k + ineg(iabs(isub_vecs(a, b))), vec2f(0.0)));
    let m : vec2f = ipow2_vec(h);
    let s : vec2f = imul_float_vec(norm_k * 0.25, m);

    return isub_vecs(imin(a,b), s);
    //return vec2f( iselect( isub_vecs(b, s), isub_vecs(a, s), ilessthan(a, b)));
}

fn opUnionInterval( s1 : vec2f, s2 : vec2f ) -> vec2f
{ 
    return imin( s1, s2 );
}
 
fn opSmoothUnionInterval( s1 : vec2f, s2 : vec2f, k : f32 ) -> vec2f
{
    return isoft_min_quadratic(s2, s1, k);
}

fn opSmoothSubtractionInterval( s1 : vec2f, s2 : vec2f, k : f32 ) -> vec2f
{
    return ineg(isoft_min_quadratic(s2, ineg(s1), k));
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

fn isign_vec2(v : vec2f) -> vec2f
{
    var s : vec2f = vec2f(1.0, 1.0);

    if (v.x < 0.0) {
        s.x = -1.0;
    }
    
    if (v.y < 0.0) {
        s.y = -1.0;
    }

    return s ; //vec2f(min(s.x, s.y), max(s.x, s.y));;
}

// Primitives

fn sphere_interval(p : mat3x3f, edit_pos : vec3f, r : f32) -> vec2f
{
	// x^2 + y^2 + z^2 - r^2
	return isub_vec_float(ilength(isub_mat_vec(p, edit_pos)), r);
}

fn cut_sphere_interval(p : mat3x3f, edit_pos : vec3f, rotation : vec4f, r : f32, h : f32) -> vec2f
{
    let pos : mat3x3f = irotate_point_quat(isub_mat_vec(p, edit_pos), rotation);

    // sampling independent computations (only depend on shape)
    var w : f32 = sqrt(r*r-h*h);

    // sampling dependant computations
    let q_x = isqrt(ipow2_vec(pos[0].xy) + ipow2_vec(pos[1].xy));
    let q_y = pos[2].xy;

    let q_x_mW = isub_vec_float(q_x, w);
    let q_y_mH = isub_vec_float(q_y, h);

    let max_0 = iadd_vecs(imul_float_vec((h-r), ipow2_vec(q_x)), imul_float_vec(w * w, ineg(isub_vec_float(imul_float_vec(2.0, q_y), (h + r)))));
    let max_1 = isub_vecs(imul_float_vec(h, q_x), imul_float_vec(w, q_y));

    let s : vec2f = imax(max_0, max_1);

    return iselect(
        iselect(
            isqrt(ipow2_vec(q_x_mW) + ipow2_vec(q_y_mH)), 
            ineg(isub_vec_float(q_y, h)),
            ilessthan(q_x, vec2f(w))
        ), 
        isub_vec_float(isqrt(ipow2_vec(q_x) + ipow2_vec(q_y)), r),
        ilessthan(s, vec2f(0.0))
    );
}

fn box_interval(p : mat3x3f, edit_pos : vec3f, rotation : vec4f, size : vec3f, r : f32) -> vec2f
{
    let mat_zero_interval : mat3x3f = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));

    // Move edit
    let interval_translated : mat3x3f = irotate_point_quat(isub_mat_vec(p, edit_pos), rotation);

    let interval_translated_sized : mat3x3f = iadd_mat_float( isub_mat_vec(iabs_mats(interval_translated), size), r);

    var i_distance : vec2f = ilength(imax_mats(interval_translated_sized, mat_zero_interval));
    var max_on_all_axis : vec2f = imax(interval_translated_sized[0].xy, imax(interval_translated_sized[1].xy, interval_translated_sized[2].xy));
    var i_dist2 : vec2f = isub_vecs(imin(max_on_all_axis , vec2f(0.0, 0.0)), vec2f(r, r));

    return iadd_vecs(i_distance, i_dist2);
}

fn capsule_interval( p : mat3x3f, c : vec3f, rotation : vec4f, radius : f32, height : f32) -> vec2f
{
    let a : mat3x3f = iavec3_vec(c);
    var pa : mat3x3f = irotate_point_quat(isub_mat_vec(p, c), rotation);

    let b : mat3x3f = iavec3_vec(c - vec3f(0.0, 0.0, height));
    var ba : mat3x3f = isub_mats(b, a);

    var h : vec2f = iclamp(idiv_vecs(idot_mat(pa, ba), idot_mat(ba, ba)), 0.0, 1.0);
    let d : mat3x3f = isub_mats(pa, imul_vec2_mat(h, ba));
    return isub_vec_float(ilength(d), radius);
}

fn cone_interval(p : mat3x3f, c : vec3f, rotation : vec4f, radius_dims : vec2f, height : f32) -> vec2f
{
    var r2 = radius_dims.x;
    var r1 = radius_dims.y;
    var h = height * 0.5;

    var pos : mat3x3f = irotate_point_quat(isub_mat_vec(p, c), rotation);

    let q_x = isqrt(ipow2_vec(pos[0].xy) + ipow2_vec(pos[1].xy));
    let q_y = pos[2].xy;

    let k1 = vec2f(r2, h);
    let k2 = vec2f(r2 - r1, 2.0 * h);

    let sr = iselect(vec2f(r2), vec2f(r1), ilessthan(q_y, vec2f(0.0)));
    let ca_x = isub_vecs(q_x, imin(q_x, sr));
    let ca_y = isub_vecs(iabs(q_y), vec2f(h));

    let m_skq_x = iavec3_vec(vec3f(isub_vecs(k1, q_x), 0.0));
    let m_skq_y = iavec3_vec(vec3f(isub_vecs(k1, q_y), 0.0));

    let mk2 : mat3x3f = iavec3_vec(vec3f(k2, 0.0));

    let cb_x = q_x - k1 + k2 * iclamp( idot_mat(m_skq_x, mk2) / idot_mat(mk2, mk2), 0.0, 1.0);
    let cb_y = q_y - k1 + k2 * iclamp( idot_mat(m_skq_y, mk2) / idot_mat(mk2, mk2), 0.0, 1.0);

    let cond : vec2<bool> = iand(ilessthan(cb_x, vec2f(0.0)), ilessthan(ca_y, vec2f(0.0)));
    let s : vec2f = select(vec2f(1.0), vec2f(-1.0), cond);

    let m_ca = iavec3_vecs(ca_x, ca_y, vec2f(0.0));
    let m_cb = iavec3_vecs(cb_x, cb_y, vec2f(0.0));

    return imul_vec2_vec2(s, isqrt(imin(idot_mat(m_ca, m_ca), idot_mat(m_cb, m_cb))));
}

fn cylinder_interval(p : mat3x3f, start_pos : vec3f, rotation : vec4f, radius : f32, height : f32) -> vec2f
{
    let cyl_origin : mat3x3f = irotate_point_quat(isub_mat_vec(p, start_pos), rotation);
    
    let d_x : vec2f = isub_vecs(iabs(isqrt(ipow2_vec(cyl_origin[0].xy) + ipow2_vec(cyl_origin[2].xy))), vec2f(radius, radius)); 
    let d_y : vec2f = isub_vecs(iabs(cyl_origin[1].xy), vec2f(height, height)); 

    let max_d_x : vec2f = imax(vec2f(0.0), d_x);
    let max_d_y : vec2f = imax(vec2f(0.0), d_y);

    return iadd_vecs(imin(imax(d_x, d_y), vec2f(0.0)), isqrt(ipow2_vec(max_d_x.xy) + ipow2_vec(max_d_y.xy)));
}

fn torus_interval( p : mat3x3f, c : vec3f, t : vec2f, rotation : vec4f) -> vec2f
{
    let pos : mat3x3f = irotate_point_quat(isub_mat_vec(p, c), rotation);

    let d_x = isub_vec_float(isqrt(ipow2_vec(pos[0].xy) + ipow2_vec(pos[2].xy)), t.x);
    let d_y = pos[1].xy;

    return isub_vec_float(isqrt(ipow2_vec(d_x) + ipow2_vec(d_y)), t.y);
}

fn capped_torus_interval( p : mat3x3f, c : vec3f, t : vec2f, rotation : vec4f, sc : vec2f) -> vec2f
{
    let pos : mat3x3f = irotate_point_quat(isub_mat_vec(p, c), rotation);

    let ra = t.x;
    let rb = t.y;
    let posX : vec2f = iabs(pos[0].xy);
    let posY : vec2f = pos[1].xy;

    let mPos = iavec3_vecs(posX, posY, pos[2].xy);

    var cond : vec2<bool> = igreaterthan( imul_float_vec(sc.y, posX), imul_float_vec(sc.x, posY));
    var lenPosXY : vec2f = isqrt(ipow2_vec(posX) + ipow2_vec(posY));
    var posXYdotSC = idot_mat(iavec3_vecs(posX, posY, vec2f(0.0)), iavec3_vecs(sc.xx, sc.yy, vec2f(0.0)));
    var k : vec2f = iselect(lenPosXY, posXYdotSC, cond);

    var pdotp : vec2f = idot_mat(mPos, mPos);

    return isub_vec_float(isqrt( isub_vec_float(isub_vecs(pdotp, imul_float_vec(2.0*ra, k)), -ra * ra) ), rb);
}

fn bezier_interval( p : mat3x3f, start : vec3f, cp : vec3f, end : vec3f, thickness : f32, rotation : vec4f) -> vec2f
{
    let b0 : mat3x3f = irotate_point_quat(isub_mats(iavec3_vec(start), p), rotation);
    let b1 : mat3x3f = irotate_point_quat(isub_mats(iavec3_vec(cp), p), rotation);
    let b2 : mat3x3f = irotate_point_quat(isub_mats(iavec3_vec(end), p), rotation);

    var b01 : mat3x3f = icross_mats(b0, b1);
    var b12 : mat3x3f = icross_mats(b1, b2);
    var b20 : mat3x3f = icross_mats(b2, b0);

    var n : mat3x3f = iadd_mats(iadd_mats(b01, b12), b20);

    var a : vec2f = ineg(idot_mat(b20, n));
    var b : vec2f = ineg(idot_mat(b01, n));
    var d : vec2f = ineg(idot_mat(b12, n));

    var m : vec2f = ineg(idot_mat(n, n));

    var g0 : mat3x3f = imul_vec2_mat(isub_vecs(d, b), b1);
    var g1 : mat3x3f = imul_vec2_mat(iadd_vecs(b, imul_float_vec(0.5, a)), b2);
    var g2 : mat3x3f = imul_vec2_mat(isub_vecs(ineg(d), imul_float_vec(0.5, a)), b0);
    var g : mat3x3f = iadd_mats(iadd_mats(g0, g1), g2);

    var f : vec2f = isub_vecs(imul_float_vec(0.25, ipow2_vec(a)), imul_vec2_vec2(b, d));
    var k : mat3x3f = iadd_mats(isub_mats(b0, imul_float_mat(2.0, b1)), b2);

    var t00 : vec2f = iadd_vecs(imul_float_vec(0.5, a), b);         // [a * 0.5 + b]
    var t01 : vec2f = imul_float_vec(0.5, f);                       // [f * 0.5]
    var t10 : vec2f = idiv_vecs(idot_mat(g,k), idot_mat(g, g));     // [dot(g,k) / dot(g,g)]
    var t1 : vec2f = isub_vecs(t00, imul_vec2_vec2(t01, t10));      // [a * 0.5 + b] - [0.5 * f * dot(g,k) / dot(g,g)]
    var t2 : vec2f = idiv_vecs(t1, m);                              // [a * 0.5 + b - 0.5 * f * dot(g,k) / dot(g,g)] / [m]
    var t : vec2f = iclamp(t2, 0.0, 1.0 );

    var x0 : mat3x3f = imix_mats(b0, b1, t);
    var x1 : mat3x3f = imix_mats(b1, b2, t);
    var x : mat3x3f = idiv_mats(x0, x1);

    return isub_vec_float(ilength(x), thickness);
}


// COMPOUND SDF FUNCTIONS
fn eval_interval_stroke_sphere_substraction( position : mat3x3f, current_surface : vec2f, curr_stroke: ptr<storage, Stroke>) -> vec2f {
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
    let edit_count : u32 = (*curr_stroke).edit_count;
    let parameters : vec4f = (*curr_stroke).parameters;

    let smooth_factor : f32 = parameters.w;

    for(var i : u32 = 0u; i < edit_count; i++) {
        let curr_edit : Edit = edit_array[i];
        let radius : f32 = curr_edit.dimensions.x;
        tmp_surface = sphere_interval(position, curr_edit.position, radius);
        result_surface = opSmoothSubtractionInterval(result_surface, tmp_surface, smooth_factor);
    }
    
    return result_surface;
}

fn eval_interval_stroke_sphere_union( position : mat3x3f, current_surface : vec2f, curr_stroke: ptr<storage, Stroke>,  dimension_margin : vec4f) -> vec2f {
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;

    let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
    let edit_count : u32 = (*curr_stroke).edit_count;
    let parameters : vec4f = (*curr_stroke).parameters;

    let smooth_factor : f32 = parameters.w;

    for(var i : u32 = 0u; i < edit_count; i++) {
        let curr_edit : Edit = edit_array[i];
        let radius : f32 = curr_edit.dimensions.x + dimension_margin.x;
        tmp_surface = sphere_interval(position, curr_edit.position, radius);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
    }
    
    return result_surface;
}


// BOX SDFS ================
fn eval_interval_stroke_box_substraction(position : mat3x3f, current_surface : vec2f, curr_stroke: ptr<storage, Stroke>) -> vec2f {
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;
    
    let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
    let edit_count : u32 = (*curr_stroke).edit_count;
    let parameters : vec4f = (*curr_stroke).parameters;

    let smooth_factor : f32 = parameters.w;

    for(var i : u32 = 0u; i < edit_count; i++) {
        let curr_edit : Edit = edit_array[i];
        var size : vec3f = curr_edit.dimensions.xyz;
        let size_param = (curr_edit.dimensions.w / 0.1) * size.x; // Make Rounding depend on the side length
        size -= size_param;

        tmp_surface = box_interval(position, curr_edit.position, curr_edit.rotation, size, size_param);
        result_surface = opSmoothSubtractionInterval(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}

fn eval_interval_stroke_box_union(position : mat3x3f, current_surface : vec2f, curr_stroke: ptr<storage, Stroke>, dimension_margin : vec4f) -> vec2f {
    var result_surface : vec2f = current_surface;
    var tmp_surface : vec2f;
    
    let edit_array : ptr<storage, array<Edit, MAX_EDITS_PER_EVALUATION>> = &((*curr_stroke).edits);
    let edit_count : u32 = (*curr_stroke).edit_count;
    let parameters : vec4f = (*curr_stroke).parameters;

    let smooth_factor : f32 = parameters.w;

    for(var i : u32 = 0u; i < edit_count; i++) {
        let curr_edit : Edit = edit_array[i];
        var size : vec3f = curr_edit.dimensions.xyz + dimension_margin.xyz;
        let size_param = (curr_edit.dimensions.w / 0.1) * size.x; // Make Rounding depend on the side length
        size -= size_param;

        tmp_surface = box_interval(position, curr_edit.position, curr_edit.rotation, size, size_param);
        result_surface = opSmoothUnionInterval(result_surface, tmp_surface, smooth_factor);
    }

    return result_surface;
}


fn evaluate_stroke_interval_2( position: mat3x3f, stroke: ptr<storage, Stroke, read>, current_surface : vec2f) -> vec2f {
    let stroke_operation : u32 = (*stroke).operation;
    let stroke_primitive : u32 = (*stroke).primitive;

    let curr_stroke_code : u32 = stroke_primitive | (stroke_operation << 4u);

    var result_surface : vec2f = current_surface;

    switch(curr_stroke_code) {
        case SD_SPHERE_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_sphere_union(position, result_surface, stroke, vec4f(0.0));
            break;
        }
        case SD_SPHERE_SMOOTH_OP_SUBSTRACTION:{
            result_surface = eval_interval_stroke_sphere_substraction(position, result_surface, stroke);
            break;
        }
        case SD_BOX_SMOOTH_OP_UNION: {
            result_surface = eval_interval_stroke_box_union(position, result_surface, stroke, vec4f(0.0));
            break;
        }
        case SD_BOX_SMOOTH_OP_SUBSTRACTION: {
            result_surface = eval_interval_stroke_box_substraction(position, result_surface, stroke);
            break;
        }
        default: {}
    }

    return result_surface;
}

fn evaluate_stroke_interval_force_union( position: mat3x3f, stroke: ptr<storage, Stroke, read>, current_surface : vec2f) -> vec2f {
    let stroke_primitive : u32 = (*stroke).primitive;
    let SMOOTH_FACTOR : f32 = (*stroke).parameters.w;

    let dimensions_extra : vec4f = vec4f(SMOOTH_FACTOR);

    var result_surface : vec2f = current_surface;

    switch(stroke_primitive) {
        case SD_SPHERE: {
            result_surface = eval_interval_stroke_sphere_union(position, result_surface, stroke, dimensions_extra);
            break;
        }
        case SD_BOX: {
            result_surface = eval_interval_stroke_box_union(position, result_surface, stroke, dimensions_extra);
            break;
        }
        default: {}
    }

    return result_surface;
}

fn eval_edit_interval( p_x : vec2f, p_y : vec2f, p_z : vec2f,  primitive : u32, operation : u32, edit_parameters : vec4f, current_interval : vec2f, edit : Edit, resulting_interval : ptr<function, vec2f>) -> vec2f
{
    var pSurface : vec2f;

    // Center in texture (position 0,0,0 is just in the middle)
    var size : vec3f = edit.dimensions.xyz;
    var radius : f32 = edit.dimensions.x;
    var size_param : f32 = edit.dimensions.w;

    // 0 no cap ... 1 fully capped
    var cap_value : f32 = edit_parameters.y;

    var onion_thickness : f32 = edit_parameters.x;
    let do_onion = onion_thickness > 0.0;

    let smooth_factor : f32 = edit_parameters.w;

    switch (primitive) {
        case SD_SPHERE: {
            onion_thickness = map_thickness( onion_thickness, radius );
            radius -= onion_thickness; // Compensate onion size
            if(cap_value > 0.0) { 
                cap_value = cap_value * 2.0 - 1.0;
                pSurface = cut_sphere_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, radius, radius * cap_value * 0.999);
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
            size -= size_param;
            size_param -= onion_thickness;

            pSurface = box_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, size, size_param);
            break;
        }
        case SD_CAPSULE: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            pSurface = capsule_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, size_param, radius);
            break;
        }
        case SD_CONE: {
            // onion_thickness = map_thickness( onion_thickness, 0.01 );
            radius = max(radius * (1.0 - cap_value), 0.0025);
            var dims = vec2f(size_param, size_param * cap_value);
            pSurface = cone_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, dims, radius);
            break;
        }
        // case SD_PYRAMID: {
        //     pSurface = sdPyramid(position, edit.position, edit.rotation, radius, size_param, edit.color);
        //     break;
        // }
        case SD_CYLINDER: {
            //onion_thickness = map_thickness( onion_thickness, size_param );
            //size_param -= onion_thickness; // Compensate onion size
            pSurface = cylinder_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.rotation, size_param, radius);
            break;
        }
        case SD_TORUS: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            size_param = clamp( size_param, 0.0001, radius );
            if(cap_value > 0.0) {
                var an = M_PI * (1.0 - cap_value);
                var angles = vec2f(sin(an), cos(an));
                pSurface = capped_torus_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, vec2f(radius, size_param), edit.rotation, angles);
            } else {
                pSurface = torus_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, vec2f(radius, size_param), edit.rotation);
            }
            break;
        }
        case SD_BEZIER: {
            var curve_thickness : f32 = 0.01;
            pSurface = bezier_interval(iavec3_vecs(p_x, p_y, p_z), edit.position, edit.position + vec3f(0.1, 0.2, 0.0), edit.position + vec3f(0.2, 0.0, 0.0), curve_thickness, edit.rotation);
            break;
        }
        default: {
            break;
        }
    }

    *resulting_interval = pSurface;

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