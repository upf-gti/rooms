
// SD Primitives

const SD_SPHERE         = 0;
const SD_BOX            = 1;
const SD_ELLIPSOID      = 2;
const SD_CONE           = 3;
const SD_PYRAMID        = 4;
const SD_CYLINDER       = 5;
const SD_CAPSULE        = 6;
const SD_TORUS          = 7;
const SD_BEZIER         = 8;

// SD Operations

const OP_UNION                  = 0;
const OP_SUBSTRACTION           = 1;
const OP_INTERSECTION           = 2;
const OP_PAINT                  = 3;
const OP_SMOOTH_UNION           = 4;
const OP_SMOOTH_SUBSTRACTION    = 5;
const OP_SMOOTH_INTERSECTION    = 6;
const OP_SMOOTH_PAINT           = 7;

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

// Material operation functions
fn Material_mult_by(m : Material, v : f32) -> Material {
    return Material(m.albedo * v, m.roughness * v, m.metalness * v);
}

fn Material_sum_Material(m1 : Material, m2 : Material) -> Material {
    return Material(m1.albedo + m2.albedo, m1.roughness + m2.roughness, m1.metalness + m2.metalness);
}

fn Material_mix(m1 : Material, m2 : Material, t : f32) -> Material {
    return Material_sum_Material(Material_mult_by(m1, 1.0 - t), Material_mult_by(m2, t));
}

// Primitives

fn sdPlane( p : vec3f, c : vec3f, n : vec3f, h : f32, material : Material) -> Surface
{
    // n must be normalized
    var sf : Surface;
    sf.distance = dot(p - c, n) + h;
    sf.material = material;
    return sf;
}

fn sdSphere( p : vec3f, c : vec3f, r : f32, material : Material) -> Surface
{
    var sf : Surface;
    sf.distance = length(p - c) - r;
    sf.material = material;
    return sf;
}

fn sdCutSphere( p : vec3f, c : vec3f, rotation : vec4f, r : f32, h : f32, material : Material) -> Surface
{
    var sf : Surface;
    // sampling independent computations (only depend on shape)
    var w = sqrt(r*r-h*h);

    let pos : vec3f = rotate_point_quat(p - c, rotation);

    // sampling dependant computations
    var q = vec2f( length(pos.xy), pos.z );
    var s = max( (h-r)*q.x*q.x+w*w*(h+r-2.0*q.y), h*q.x-w*q.y );
    if(s<0.0) {
        sf.distance = length(q) - r;
    } else if(q.x<w) {
        sf.distance = h - q.y;
    } else  {
        sf.distance = length(q-vec2f(w,h));
    }
                    
    sf.material = material;
    return sf;
}

fn sdBox( p : vec3f, c : vec3f, rotation : vec4f, s : vec3f, r : f32, material : Material) -> Surface
{
    var sf : Surface;

    let pos : vec3f = rotate_point_quat(p - c, rotation);

    let q : vec3f = abs(pos) - s;
    sf.distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    sf.material = material;
    return sf;
}

fn sdCapsule( p : vec3f, a : vec3f, b : vec3f, rotation : vec4f, r : f32, material : Material) -> Surface
{
    var sf : Surface;
    let posA : vec3f = rotate_point_quat(p - a, rotation);

    let pa : vec3f = posA;
    let ba : vec3f = b - a;
    let h : f32 = clamp(dot(pa,ba) / dot(ba, ba), 0.0, 1.0);
    sf.distance = length(pa-ba*h) - r;
    sf.material = material;
    return sf;
}

// t: (base radius, top radius)
fn sdCone( p : vec3f, a : vec3f, height : f32, rotation : vec4f, t : vec2f, material : Material) -> Surface
{
    var sf : Surface;
    var r2 = t.x;
    var r1 = t.y;

    let pos : vec3f = rotate_point_quat(p - a, rotation) + vec3f(0.0, 0.0, height);
    let q = vec2f( length(pos.xy), pos.z );
    let k1 = vec2f(r2, height);
    let k2 = vec2f(r2-r1, 2.0 * height);
    let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y<0.0)), abs(q.y) - height);
    let cb = q - k1 + k2 * clamp( dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s : f32 = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);

    sf.distance = s * sqrt(min(dot(ca, ca), dot(cb, cb)));
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

fn sdCylinder(p : vec3f, a : vec3f, rotation : vec4f, r : f32, h : f32, rr : f32, material : Material) -> Surface
{
    var sf : Surface;

    let posA : vec3f = rotate_point_quat(p - a, rotation);

    let d : vec2f = abs(vec2f(length(vec2f(posA.x, posA.y)), posA.z)) - vec2(r, h);
    sf.distance = min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - rr;
    sf.material = material;
    return sf;
}

// t: (circle radius, thickness radius)
fn sdTorus( p : vec3f, c : vec3f, t : vec2f, rotation : vec4f, material : Material) -> Surface
{
    var sf : Surface;
    let pos : vec3f = rotate_point_quat(p - c, rotation);
    var q = vec2f(length(pos.xy) - t.x, pos.z);
    sf.distance = length(q) - t.y;
    sf.material = material;
    return sf;
}

// t: (circle radius, thickness radius)
fn sdCappedTorus( p : vec3f, c : vec3f, t : vec2f, rotation : vec4f, sc : vec2f, material : Material) -> Surface
{
    var sf : Surface;
    var pos : vec3f = rotate_point_quat(p - c, rotation);

    let ra = t.x;
    let rb = t.y;
    pos.x = abs(pos.x);
    var k = select(length(pos.xy), dot(pos.xy,sc), sc.y*pos.x > sc.x*pos.y);

    sf.distance = sqrt( dot(pos,pos) + ra*ra - 2.0*ra*k ) - rb;
    sf.material = material;
    return sf;
}

// IQ adaptation to 3d of http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
// { dist, t, y (above the plane of the curve, x (away from curve in the plane of the curve))
fn sdQuadraticBezier(p : vec3f, start : vec3f, cp : vec3f, end : vec3f, thickness : f32, rotation : vec4f, material : Material) -> Surface
{
    var b0 : vec3f = start - p;
    var b1 : vec3f = cp - p;
    var b2 : vec3f = end - p;
    
    var b01 : vec3f = cross(b0, b1);
    var b12 : vec3f = cross(b1, b2);
    var b20 : vec3f = cross(b2, b0);
    
    var n : vec3f =  b01 + b12 + b20;
    
    var a : f32 = -dot(b20, n);
    var b : f32 = -dot(b01, n);
    var d : f32 = -dot(b12, n);

    var m : f32 = -dot(n,n);
    
    var g : vec3f =  (d-b)*b1 + (b+a*0.5)*b2 + (-d-a*0.5)*b0;
    var f : f32 = a*a*0.25-b*d;
    var k : vec3f = b0-2.0*b1+b2;
    var t : f32 = clamp((a*0.5+b-0.5*f*dot(g,k)/dot(g,g))/m, 0.0, 1.0 );
    
    var sf : Surface;
    sf.distance = length(mix(mix(b0,b1,t), mix(b1,b2,t),t)) - thickness;
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
    let m : f32 = h * h * h * 0.5;
    return vec2f(min(a, b) - h * h * k * 0.25, select(1.0 - m, m, a < b)); 
}

// From iqulzes and Dreams
fn sminPoly(a : f32, b : f32, k : f32) -> vec2f {
    let h : f32 = max(k - abs(a - b), 0.0) / k;
    let m : f32 = h*h;
    let s : f32 = m*k*(1.0/4.0);

    if (a < b) {
        return vec2f(a - s, m);
    } else {
        return vec2f(b - s, 1.0 - m);
    }
}

fn opSmoothUnion( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let smin : vec2f = soft_min(s2.distance, s1.distance, k);
    //let smin : vec2f = sminPoly(s2.distance, s1.distance, k);
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

fn opPaint( s1 : Surface, s2 : Surface, material : Material ) -> Surface
{
    var sColorInter : Surface = opIntersection(s1, s2);
    sColorInter.material = material;
    return opUnion(s1, sColorInter);
}

fn opSmoothSubtraction( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let smin : vec2f = soft_min(s2.distance, -s1.distance, k);
    var s : Surface;
    s.distance = -smin.x;
    s.material = s2.material;
    return s;
}

fn opSmoothIntersection( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let h : f32 = max(k - abs(s1.distance - s2.distance), 0.0);
    var s : Surface;
    s.distance = max(s1.distance, s2.distance) + h * h * 0.25 / k;
    //s.color = s1.color;
    return s;
}

fn opSmoothPaint( s1 : Surface, s2 : Surface, material : Material, k : f32 ) -> Surface
{
    var sColorInter : Surface = opIntersection(s1, s2);
    sColorInter.material = material;
    let u : Surface = opUnion(s1, sColorInter);
    var s : Surface = opSmoothUnion(s1, sColorInter, k);
    s.distance = u.distance;
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

fn evaluate_edit( position : vec3f, primitive : u32, operation : u32, parameters : vec4f, current_surface : Surface, stroke_material : Material, edit : Edit) -> Surface
{
    var pSurface : Surface;

    // Center in texture (position 0,0,0 is just in the middle)
    var size : vec3f = edit.dimensions.xyz;
    var radius : f32 = edit.dimensions.x;
    var size_param : f32 = edit.dimensions.w;

    // 0 no cap ... 1 fully capped
    var cap_value : f32 = parameters.y;

    var onion_thickness : f32 = parameters.x;
    let do_onion = onion_thickness > 0.0;

    let smooth_factor : f32 = parameters.w;

    switch (primitive) {
        case SD_SPHERE: {
            onion_thickness = map_thickness( onion_thickness, radius );
            radius -= onion_thickness; // Compensate onion size
            if(cap_value > 0.0) { 
                cap_value = cap_value * 2.0 - 1.0;
                pSurface = sdCutSphere(position, edit.position, edit.rotation, radius, radius * cap_value * 0.999, stroke_material);
            } else {
                pSurface = sdSphere(position, edit.position, radius, stroke_material);
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

            pSurface = sdBox(position, edit.position, edit.rotation, size, size_param, stroke_material);
            break;
        }
        case SD_CAPSULE: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            var height = radius; // ...
            pSurface = sdCapsule(position, edit.position, edit.position - vec3f(0.0, 0.0, height), edit.rotation, size_param, stroke_material);
            break;
        }
        case SD_CONE: {
            onion_thickness = map_thickness( onion_thickness, 0.01 );
            radius = max(radius * (1.0 - cap_value), 0.0025);
            var dims = vec2f(size_param, size_param * cap_value);
            pSurface = sdCone(position, edit.position, radius, edit.rotation, dims, stroke_material);
            break;
        }
        // case SD_PYRAMID: {
        //     pSurface = sdPyramid(position, edit.position, edit.rotation, radius, size_param, edit_color);
        //     break;
        // }
        case SD_CYLINDER: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            pSurface = sdCylinder(position, edit.position, edit.rotation, size_param, radius, 0.0, stroke_material);
            break;
        }
        case SD_TORUS: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            size_param = clamp( size_param, 0.0001, radius );
            if(cap_value > 0.0) {
                var an = M_PI * (1.0 - cap_value);
                var angles = vec2f(sin(an), cos(an));
                pSurface = sdCappedTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, angles, stroke_material);
            } else {
                pSurface = sdTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, stroke_material);
            }
            break;
        }
        case SD_BEZIER: {
            var curve_thickness : f32 = 0.01;
            pSurface = sdQuadraticBezier(position, edit.position, edit.position + vec3f(0.0, 0.1, 0.0), edit.position + vec3f(0.2, 0.1, 0.0), curve_thickness, edit.rotation, stroke_material);
            break;
        }
        default: {
            break;
        }
    }

    // Shape edition ...
    if( do_onion && (operation == OP_UNION || operation == OP_SMOOTH_UNION) )
    {
        pSurface = opOnion(pSurface, onion_thickness);
    }

    switch (operation) {
        case OP_UNION: {
            pSurface = opUnion(current_surface, pSurface);
            break;
        }
        case OP_SUBSTRACTION:{
            pSurface = opSubtraction(current_surface, pSurface);
            break;
        }
        case OP_INTERSECTION: {
            pSurface = opIntersection(current_surface, pSurface);
            break;
        }
        case OP_PAINT: {
            pSurface = opPaint(current_surface, pSurface, stroke_material);
            break;
        }
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
            pSurface = opSmoothPaint(current_surface, pSurface, stroke_material, smooth_factor);
            break;
        }
        default: {
            break;
        }
    }

    return pSurface;
}
