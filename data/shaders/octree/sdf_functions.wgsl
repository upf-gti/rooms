const M_PI = 3.14159265359;

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

const OFFSET_LUT : array<vec3f, 8> = array<vec3f, 8>(
    vec3f(-1.0, -1.0, -1.0),
    vec3f( 1.0, -1.0, -1.0),
    vec3f(-1.0,  1.0, -1.0),
    vec3f( 1.0,  1.0, -1.0),
    vec3f(-1.0, -1.0,  1.0),
    vec3f( 1.0, -1.0,  1.0),
    vec3f(-1.0,  1.0,  1.0),
    vec3f( 1.0,  1.0,  1.0)
);

// Data containers

struct Surface {
    color    : vec3f,
    distance : f32
};

// Primitives

fn rotate_point_quat(position : vec3f, rotation : vec4f) -> vec3f
{
    return position + 2.0 * cross(rotation.xyz, cross(rotation.xyz, position) + rotation.w * position);
}

fn quat_mult(q1 : vec4f, q2 : vec4f) -> vec4f
{
    return vec4f(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

fn quat_conj(q : vec4f) -> vec4f
{
    return vec4f(-q.x, -q.y, -q.z, q.w);
}

fn quat_inverse(q : vec4f) -> vec4f
{
    let conj : vec4f = quat_conj(q);
    return conj / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

fn sdPlane( p : vec3f, c : vec3f, n : vec3f, h : f32, color : vec3f ) -> Surface
{
    // n must be normalized
    var sf : Surface;
    sf.distance = dot(p - c, n) + h;
    sf.color = color;
    return sf;
}

fn sdSphere( p : vec3f, c : vec3f, r : f32, color : vec3f) -> Surface
{
    var sf : Surface;
    sf.distance = length(p - c) - r;
    sf.color = color;
    return sf;
}

fn sdCutSphere( p : vec3f, c : vec3f, rotation : vec4f, r : f32, h : f32, color : vec3f ) -> Surface
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
                    
    sf.color = color;
    return sf;
}

fn sdBox( p : vec3f, c : vec3f, rotation : vec4f, s : vec3f, r : f32, color : vec3f ) -> Surface
{
    var sf : Surface;

    let pos : vec3f = rotate_point_quat(p - c, rotation);

    let q : vec3f = abs(pos) - s;
    sf.distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    sf.color = color;
    return sf;
}

fn sdCapsule( p : vec3f, a : vec3f, b : vec3f, rotation : vec4f, r : f32, color : vec3f ) -> Surface
{
    var sf : Surface;
    let posA : vec3f = rotate_point_quat(p - a, rotation);

    let pa : vec3f = posA;
    let ba : vec3f = b - a;
    let h : f32 = clamp(dot(pa,ba) / dot(ba, ba), 0.0, 1.0);
    sf.distance = length(pa-ba*h) - r;
    sf.color = color;
    return sf;
}

// t: (base radius, top radius)
fn sdCone( p : vec3f, a : vec3f, b : vec3f, rotation : vec4f, t : vec2f, color : vec3f ) -> Surface
{
    var sf : Surface;
    var ra = t.x;
    var rb = t.y;

    let pa : vec3f = rotate_point_quat(p - a, rotation);
    let ba = b - a;
    var rba  = rb-ra;
    var baba = dot(ba,ba);
    var papa = dot(pa,pa);
    var paba = dot(pa,ba)/baba;

    var x = sqrt( papa - paba*paba*baba );

    var cax = max(0.0,x-select(rb, ra, paba < 0.5));
    var cay = abs(paba-0.5) - 0.5;

    var k = rba*rba + baba;
    var f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );

    var cbx = x-ra - f*rba;
    var cby = paba - f;
    
    var s = select(1.0, -1.0, cbx < 0.0 && cay < 0.0);
    
    sf.distance = s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
    sf.color = color;
    return sf;
}

fn sdPyramid( p : vec3f, c : vec3f, rotation : vec4f, r : f32, h : f32, color : vec3f ) -> Surface
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
    sf.color = color;
    return sf;
}

fn sdCylinder(p : vec3f, a : vec3f, b : vec3f, rotation : vec4f, r : f32, rr : f32, color : vec3f) -> Surface
{
    var sf : Surface;

    let posA : vec3f = rotate_point_quat(p - a, rotation);

    let pa : vec3f = posA;
    let ba : vec3f = b - a;
    let baba : f32 = dot(ba, ba);
    let paba : f32 = dot(pa, ba);

    let x  : f32 = length(pa * baba - ba * paba) - r * baba;
    let y  : f32 = abs(paba - baba * 0.5) - baba * 0.5;
    let x2 : f32 = x * x;
    let y2 : f32 = y * y * baba;
    let d  : f32 = select(select(0.0, x2, x > 0.0) + select(0.0, y2, y > 0.0), -min(x2, y2), max(x, y) < 0.0);

    sf.distance = sign(d) * sqrt(abs(d)) / baba - rr;
    sf.color = color;
    return sf;
}

// t: (circle radius, thickness radius)
fn sdTorus( p : vec3f, c : vec3f, t : vec2f, rotation : vec4f, color : vec3f ) -> Surface
{
    var sf : Surface;
    let pos : vec3f = rotate_point_quat(p - c, rotation);
    var q = vec2f(length(pos.xy) - t.x, pos.z);
    sf.distance = length(q) - t.y;
    sf.color = color;
    return sf;
}

// t: (circle radius, thickness radius)
fn sdCappedTorus( p : vec3f, c : vec3f, t : vec2f, rotation : vec4f, sc : vec2f, color : vec3f ) -> Surface
{
    var sf : Surface;
    var pos : vec3f = rotate_point_quat(p - c, rotation);

    let ra = t.x;
    let rb = t.y;
    pos.x = abs(pos.x);
    var k = select(length(pos.xy), dot(pos.xy,sc), sc.y*pos.x > sc.x*pos.y);

    sf.distance = sqrt( dot(pos,pos) + ra*ra - 2.0*ra*k ) - rb;
    sf.color = color;
    return sf;
}

// IQ adaptation to 3d of http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
// { dist, t, y (above the plane of the curve, x (away from curve in the plane of the curve))
fn sdQuadraticBezier(p : vec3f, start : vec3f, cp : vec3f, end : vec3f, thickness : f32, rotation : vec4f, color : vec3f) -> Surface
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
    sf.color = color;
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
    sf.color = mix(s2.color, s1.color, smin.y);
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
    s.color = s1.color;
    return s;
}

fn opPaint( s1 : Surface, s2 : Surface, paintColor : vec3f ) -> Surface
{
    var sColorInter : Surface = opIntersection(s1, s2);
    sColorInter.color = paintColor;
    return opUnion(s1, sColorInter);
}

fn opSmoothSubtraction( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let smin : vec2f = soft_min(s2.distance, -s1.distance, k);
    var s : Surface;
    s.distance = -smin.x;
    s.color = s1.color;
    return s;
}

fn opSmoothIntersection( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let h : f32 = max(k - abs(s1.distance - s2.distance), 0.0);
    var s : Surface;
    s.distance = max(s1.distance, s2.distance) + h * h * 0.25 / k;
    s.color = s1.color;
    return s;
}

fn opSmoothPaint( s1 : Surface, s2 : Surface, paintColor : vec3f, k : f32 ) -> Surface
{
    var sColorInter : Surface = opIntersection(s1, s2);
    sColorInter.color = paintColor;
    let u : Surface = opUnion(s1, sColorInter);
    var s : Surface = opSmoothUnion(s1, sColorInter, k);
    s.distance = u.distance;
    return s;
}

fn opOnion( s1 : Surface, t : f32 ) -> Surface
{
    var s : Surface;
    s.distance = abs(s1.distance) - t;
    s.color = s1.color;
    return s;
}

fn map_thickness( t : f32, v_max : f32 ) -> f32
{
    return select( 0.0, max(t * v_max * 0.375, 0.003), t > 0.0);
}

fn evalEdit( position : vec3f, current_surface : Surface, edit : Edit, current_edit_surface : ptr<function, Surface> ) -> Surface
{
    var pSurface : Surface;

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
                pSurface = sdCutSphere(position, edit.position, edit.rotation, radius, radius * cap_value * 0.999, edit.color);
            } else {
                pSurface = sdSphere(position, edit.position, radius, edit.color);
            }
            break;
        }
        case SD_BOX: {
            onion_thickness = map_thickness( onion_thickness, size.x );
            size_param = (size_param / 0.1) * size.x; // Make Rounding depend on the side length

            // Compensate onion size (Substract from box radius bc onion will add it later...)
            size -= onion_thickness;
            size_param -= onion_thickness; 

            pSurface = sdBox(position, edit.position, edit.rotation, size - size_param, size_param, edit.color);
            break;
        }
        case SD_CAPSULE: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            var height = radius; // ...
            pSurface = sdCapsule(position, edit.position, edit.position - vec3f(0.0, 0.0, height), edit.rotation, size_param, edit.color);
            break;
        }
        case SD_CONE: {
            onion_thickness = map_thickness( onion_thickness, 0.01 );
            cap_value = cap_value * 0.5 + 0.5;
            radius = max(radius * (1.0 - cap_value), 0.0025);
            var dims = vec2f(size_param, size_param * cap_value);
            pSurface = sdCone(position, edit.position,  edit.position - vec3f(0.0, 0.0, radius), edit.rotation, dims, edit.color);
            break;
        }
        // case SD_PYRAMID: {
        //     pSurface = sdPyramid(position, edit.position, edit.rotation, radius, size_param, edit.color);
        //     break;
        // }
        case SD_CYLINDER: {
            onion_thickness = map_thickness( onion_thickness, size_param );
            size_param -= onion_thickness; // Compensate onion size
            pSurface = sdCylinder(position, edit.position,  edit.position - vec3f(0.0, 0.0, radius), edit.rotation, size_param, 0.0, edit.color);
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
                pSurface = sdCappedTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, angles, edit.color);
            } else {
                pSurface = sdTorus(position, edit.position, vec2f(radius, size_param), edit.rotation, edit.color);
            }
            break;
        }
        case SD_BEZIER: {
            var curve_thickness : f32 = 0.01;
            pSurface = sdQuadraticBezier(position, edit.position, edit.position + vec3f(0.0, 0.1, 0.0), edit.position + vec3f(0.2, 0.1, 0.0), curve_thickness, edit.rotation, edit.color);
            break;
        }
        default: {
            break;
        }
    }

    // Shape edition ...
    if( do_onion && (edit.operation == OP_UNION || edit.operation == OP_SMOOTH_UNION) )
    {
        pSurface = opOnion(pSurface, onion_thickness);
    }

    *current_edit_surface = pSurface;

    switch (edit.operation) {
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
            pSurface = opPaint(current_surface, pSurface, edit.color);
            break;
        }
        case OP_SMOOTH_UNION: {
            pSurface = opSmoothUnion(current_surface, pSurface, SMOOTH_FACTOR);
            break;
        }
        case OP_SMOOTH_SUBSTRACTION: {
            pSurface = opSmoothSubtraction(current_surface, pSurface, SMOOTH_FACTOR);
            break;
        }
        case OP_SMOOTH_INTERSECTION: {
            pSurface = opSmoothIntersection(current_surface, pSurface, SMOOTH_FACTOR);
            break;
        }
        case OP_SMOOTH_PAINT: {
            pSurface = opSmoothPaint(current_surface, pSurface, edit.color, SMOOTH_FACTOR);
            break;
        }
        default: {
            break;
        }
    }

    return pSurface;
}
