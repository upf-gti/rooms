// https://www.shadertoy.com/view/4djSRW
fn hash33( p3 : vec3f ) -> vec3f
{
	var p : vec3f = fract(p3 * vec3f(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// https://www.shadertoy.com/view/slX3D2
fn hash33u( p3 : vec3u ) -> vec3f
{
    var x : u32 = p3.x;
    var y : u32 = p3.y;
    var z : u32 = p3.z;

    // Pick some enthropy source values.
    // Try different values.
    let enthropy0 : u32 = 1200u;
    let enthropy1 : u32 = 4500u;
    let enthropy2 : u32 = 6700u;
    let enthropy3 : u32 = 8900u;

    // Use linear offset method to mix coordinates.
    var value0 : u32 = z * enthropy3 * enthropy2 + y * enthropy2 + x;
    var value1 : u32 = y * enthropy3 * enthropy2 + x * enthropy2 + z;
    var value2 : u32 = x * enthropy3 * enthropy2 + z * enthropy2 + y;

    // Calculate hash.
	value0 += enthropy1; value0 *= 445593459u; value0 ^= enthropy0;
    value1 += enthropy1; value1 *= 445593459u; value1 ^= enthropy0;
    value2 += enthropy1; value2 *= 445593459u; value2 ^= enthropy0;

    // 2.0f / 4294967295.0f = 4.6566128730773926e-10

    return vec3f(
        f32(value0 * value0 * value0) * 4.6566128730773926e-10 - 1.0,
        f32(value1 * value1 * value1) * 4.6566128730773926e-10 - 1.0,
        f32(value2 * value2 * value2) * 4.6566128730773926e-10 - 1.0);
}

// https://www.shadertoy.com/view/Ms2GDc
fn hash3_sin( p3 : vec3f ) -> vec3f
{
	var p : vec3f = vec3f( dot(p3, vec3f(127.1, 311.7, 213.6)),
			  dot(p3, vec3f(327.1, 211.7, 113.6)),
			  dot(p3, vec3f(269.5, 183.3, 351.1)) );
	return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// https://www.shadertoy.com/view/slX3D2
fn perlin_noise_3d( p : vec3f ) -> f32
{
	// Position in grid (fractal part)
	var i : vec3f = floor( p );

	// Offset in position (integer part)
	var f : vec3f = p - i;

	// Quintic interpolation
	var u : vec3f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

	// Trilinear Interpolation

    let g0 : vec3f = hash3_sin( (i) );
    let g1 : vec3f = hash3_sin( (i + vec3f(1.0, 0.0, 0.0)) );
    let g2 : vec3f = hash3_sin( (i + vec3f(0.0, 1.0, 0.0)) );
    let g3 : vec3f = hash3_sin( (i + vec3f(1.0, 1.0, 0.0)) );
    let g4 : vec3f = hash3_sin( (i + vec3f(0.0, 0.0, 1.0)) );
    let g5 : vec3f = hash3_sin( (i + vec3f(1.0, 0.0, 1.0)) );
    let g6 : vec3f = hash3_sin( (i + vec3f(0.0, 1.0, 1.0)) );
    let g7 : vec3f = hash3_sin( (i + vec3f(1.0, 1.0, 1.0)) );

    let d0 : vec3f = f;
    let d1 : vec3f = f - vec3f(1.0, 0.0, 0.0);
    let d2 : vec3f = f - vec3f(0.0, 1.0, 0.0);
    let d3 : vec3f = f - vec3f(1.0, 1.0, 0.0);
    let d4 : vec3f = f - vec3f(0.0, 0.0, 1.0);
    let d5 : vec3f = f - vec3f(1.0, 0.0, 1.0);
    let d6 : vec3f = f - vec3f(0.0, 1.0, 1.0);
    let d7 : vec3f = f - vec3f(1.0, 1.0, 1.0);

	let dot0 : f32 = g0.x * d0.x + g0.y * d0.y + g0.z * d0.z;
    let dot1 : f32 = g1.x * d1.x + g1.y * d1.y + g1.z * d1.z;
    let dot2 : f32 = g2.x * d2.x + g2.y * d2.y + g2.z * d2.z;
    let dot3 : f32 = g3.x * d3.x + g3.y * d3.y + g3.z * d3.z;
    let dot4 : f32 = g4.x * d4.x + g4.y * d4.y + g4.z * d4.z;
    let dot5 : f32 = g5.x * d5.x + g5.y * d5.y + g5.z * d5.z;
    let dot6 : f32 = g6.x * d6.x + g6.y * d6.y + g6.z * d6.z;
    let dot7 : f32 = g7.x * d7.x + g7.y * d7.y + g7.z * d7.z;

	return
        dot0 * (1.0 - u.x) * (1.0 - u.y) * (1.0 - u.z) +
        dot1 * u.x         * (1.0 - u.y) * (1.0 - u.z) +
        dot2 * (1.0 - u.x) * u.y         * (1.0 - u.z) +
        dot3 * u.x         * u.y         * (1.0 - u.z) +
        dot4 * (1.0 - u.x) * (1.0 - u.y) * u.z +
        dot5 * u.x         * (1.0 - u.y) * u.z +
        dot6 * (1.0 - u.x) * u.y         * u.z +
        dot7 * u.x         * u.y         * u.z;
}

// Fractional Brownian Motion, generates fractal noise
// https://github.com/PZerua/tfg/blob/master/data/shaders

fn fbm( coords : vec3f, offset : vec3f, f : f32, a : f32, o : u32 ) -> f32
{
    var n : f32 = 0.0;
    var uv : vec3f = coords + offset; // Apply offset to generation
    var amplitude : f32 = a;

    uv *= f; // Apply frequency

    for (var i : u32 = 0; i < o; i++) {
        n += amplitude * perlin_noise_3d( uv ); 
        uv = 2.0 * uv;
        amplitude /= 2.0;
    }

    return n;
}