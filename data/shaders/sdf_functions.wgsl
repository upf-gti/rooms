
// SD Primitives

const SD_SPHERE         = 0;
const SD_BOX            = 1;
const SD_ELLIPSOID      = 2;
const SD_CONE           = 3;
const SD_PYRAMID        = 4;
const SD_CYLINDER       = 5;
const SD_CAPSULE        = 6;
const SD_TORUS          = 7;
const SD_CAPPED_TORUS   = 8;

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

struct Surface {
    color    : vec3f,
    distance : f32
};

struct Edit {
    position   : vec3f,
    primitive  : u32,
    color      : vec3f,
    operation  : u32,
    size       : vec3f,
    radius     : f32
};

struct Edits {
    data : array<Edit, 1024>
}

// Primitives

fn sdPlane( p : vec3f, c : vec3f, n : vec3f, h : f32, color : vec3f ) -> Surface
{
    // n must be normalized
    var sf : Surface;
    sf.distance = dot(p - c, n) + h;
    sf.color = color;
    return sf;
}

fn sdSphere( p : vec3f, c : vec3f, s : f32, color : vec3f) -> Surface
{
    var sf : Surface;
    sf.distance = length(p - c) - s;
    sf.color = color;
    return sf;
}

fn sdBox( p : vec3f, c : vec3f, s : vec3f, r : f32, color : vec3f ) -> Surface
{
    var sf : Surface;
    let q : vec3f = abs(p - c) - s;
    sf.distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    sf.color = color;
    return sf;
}

fn sdCapsule( p : vec3f, a : vec3f, b : vec3f, r : f32, color : vec3f ) -> Surface {
    var sf : Surface;

    let pa : vec3f = p - a;
    let ba : vec3f = b - a;

    let h : f32 = clamp(dot(pa,ba) / dot(ba, ba), 0.0, 1.0);

    sf.distance = length(pa-ba*h) - r;
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

fn opSmoothUnion( s1 : Surface, s2 : Surface, k : f32 ) -> Surface
{
    let smin : vec2f = sminN(s2.distance, s1.distance, k, 3.0);
    var sf : Surface;
    sf.distance = smin.x;
    sf.color = colorMix(s2.color, s1.color, smin.y);
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
    s.color = s1.color;
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
    let smin : vec2f = sminN(s2.distance, -s1.distance, k, 2.0);
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

fn evalEdit( position : vec3f, current_surface : Surface, edit : Edit ) -> Surface
{
    var pSurface : Surface;

    const smooth_factor = 0.01;

    // Center in texture (position 0,0,0 is just in the middle)
    let offsetPosition : vec3f = edit.position + vec3f(0.5);

    switch (edit.primitive) {
        case SD_SPHERE: {
            pSurface = sdSphere(vec3f(position) / vec3f(512.0), offsetPosition, edit.radius, edit.color);
            break;
        }
        case SD_BOX: {
            pSurface = sdBox(vec3f(position) / vec3f(512.0), offsetPosition, edit.size, edit.radius, edit.color);
            break;
        }
        case SD_CAPSULE: {
            pSurface = sdCapsule(vec3f(position) / vec3f(512.0), offsetPosition, edit.size + vec3f(0.5), edit.radius, edit.color);
            break;
        }
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
            pSurface = opSmoothPaint(current_surface, pSurface, edit.color, smooth_factor);
            break;
        }
        default: {
            break;
        }
    }

    return pSurface;
}