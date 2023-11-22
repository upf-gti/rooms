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

fn imul_vec_mat(a : vec3f, b : mat3x3f) -> mat3x3f
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
    return imax( s1, ineg(s2) );
}

fn sphere_interval(p : mat3x3f, offset : vec3f, r : f32) -> vec2f
{
	// x^2 + y^2 + z^2 - r^2
	return isub_vecs(ilensq(isub_mat_vec(p, offset)), vec2f(r*r));
}

fn eval_edit_interval( p_x : vec2f, p_y : vec2f, p_z : vec2f, current_interval : vec2f, edit : Edit, current_edit_interval : ptr<function, vec2f>) -> vec2f
{
    var pSurface : vec2f;

    // Center in texture (position 0,0,0 is just in the middle)
    var size : vec3f = edit.dimensions.xyz;
    var radius : f32 = edit.dimensions.x;
    var size_param : f32 = edit.dimensions.w;
    var cap_value : f32 = edit.parameters.y;

    var onion_thickness : f32 = edit.parameters.x;
    let do_onion = onion_thickness > 0.0;

    switch (edit.primitive) {
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

            // pSurface = sdBox(position, edit.position, edit.rotation, size - size_param, size_param, edit.color);
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
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            // pSurface = sdCylinder(position, edit.position,  edit.position - vec3f(0.0, 0.0, radius), edit.rotation, size_param, 0.0, edit.color);
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
    if( do_onion && (edit.operation == OP_UNION || edit.operation == OP_SMOOTH_UNION) )
    {
        // pSurface = opOnion(pSurface, onion_thickness);
    }

    *current_edit_interval = pSurface;

    switch (edit.operation) {
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
            pSurface = opSmoothUnionInterval(current_interval, pSurface, SMOOTH_FACTOR);
            break;
        }
        case OP_SMOOTH_SUBSTRACTION: {
            pSurface = opSmoothSubtractionInterval(current_interval, pSurface, SMOOTH_FACTOR);
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