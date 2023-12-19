
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39

fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

// https://github.com/PZerua/tfg/blob/master/data/shaders

fn hash( x : vec2f ) -> vec2f
{
    let k : vec2f = vec2f( 0.3183099, 0.3678794 );
    var xx = x * k + k.yx;
    return -1.0 + 2.0 * fract( 16.0 * k * fract( xx.x * xx.y * ( xx.x + xx.y ) ) );
}

fn perlin_noise( p : vec2f ) -> f32
{
	// Position in grid
	var i : vec2f = floor( p );
	// Offset in position
	var f : vec2f = fract( p );

	// Quintic interpolation
	var u : vec2f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

	// Interpolate in x axis
	var a : f32 = mix(dot(hash(i + vec2f(0.0, 0.0)), f - vec2f(0.0, 0.0)), dot(hash(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0)), u.x);
	var b : f32 = mix(dot(hash(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0)), dot(hash(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0)), u.x);

	// Interpolate in y axis
	return mix(a, b, u.y);
}

// Fractional Brownian Motion, generates fractal noise
fn fbm( coords : vec2f, offset : vec2f, f : f32, a : f32, o : u32 ) -> f32
{
    var n : f32 = 0.0;
    var uv : vec2f = coords + offset; // Apply offset to generation
    var amplitude : f32 = a;

    uv *= f; // Apply frequency

    for (var i : u32 = 0; i < o; i++) {
        n += amplitude * perlin_noise( uv ); 
        uv = 2.0 * uv;
        amplitude /= 2.0;
    }

    return n;
}