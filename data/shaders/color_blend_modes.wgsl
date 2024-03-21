// https://www.shadertoy.com/view/XdS3RW
// from @ben

fn darken( s : vec3f, d : vec3f ) -> vec3f
{
	return min(s,d);
}

fn lighten( s : vec3f, d : vec3f ) -> vec3f
{
	return max(s,d);
}

fn multiply( s : vec3f, d : vec3f ) -> vec3f
{
	return s*d;
}

fn screen( s : vec3f, d : vec3f ) -> vec3f
{
	return s + d - s * d;
}

fn colorBurn( s : vec3f, d : vec3f ) -> vec3f
{
	return 1.0 - (1.0 - d) / s;
}

fn colorDodge( s : vec3f, d : vec3f ) -> vec3f
{
	return d / (1.0 - s);
}

fn additive( s : vec3f, d : vec3f ) -> vec3f
{
	return s + d;
}

fn linearBurn( s : vec3f, d : vec3f ) -> vec3f
{
	return s + d - 1.0;
}

fn darkerColor( s : vec3f, d : vec3f ) -> vec3f
{
	return select(d, s, s.x + s.y + s.z < d.x + d.y + d.z);
}

fn lighterColor( s : vec3f, d : vec3f ) -> vec3f
{
	return select(d, s, s.x + s.y + s.z > d.x + d.y + d.z);
}

fn foverlay( s : f32, d : f32 ) -> f32
{
	return select(1.0 - 2.0 * (1.0 - s) * (1.0 - d), 2.0 * s * d, d < 0.5);
}

fn overlay( s : vec3f, d : vec3f ) -> vec3f
{
	var c : vec3f;
	c.x = foverlay(s.x,d.x);
	c.y = foverlay(s.y,d.y);
	c.z = foverlay(s.z,d.z);
	return c;
}

// fn softLight( s : f32, d : f32 ) -> f32
// {
// 	return (s < 0.5) ? d - (1.0 - 2.0 * s) * d * (1.0 - d) 
// 		: (d < 0.25) ? d + (2.0 * s - 1.0) * d * ((16.0 * d - 12.0) * d + 3.0) 
// 					 : d + (2.0 * s - 1.0) * (sqrt(d) - d);
// }

// fn softLight( s : vec3f, d : vec3f ) -> vec3f
// {
// 	var c : vec3f;
// 	c.x = softLight(s.x,d.x);
// 	c.y = softLight(s.y,d.y);
// 	c.z = softLight(s.z,d.z);
// 	return c;
// }

fn fhardLight( s : f32, d : f32 ) -> f32
{
	return select(1.0 - 2.0 * (1.0 - s) * (1.0 - d), 2.0 * s * d, s < 0.5);
}

fn hardLight( s : vec3f, d : vec3f ) -> vec3f
{
	var c : vec3f;
	c.x = fhardLight(s.x,d.x);
	c.y = fhardLight(s.y,d.y);
	c.z = fhardLight(s.z,d.z);
	return c;
}

fn fvividLight( s : f32, d : f32 ) -> f32
{
	return select(d / (2.0 * (1.0 - s)), 1.0 - (1.0 - d) / (2.0 * s), s < 0.5);
}

fn vividLight( s : vec3f, d : vec3f ) -> vec3f
{
	var c : vec3f;
	c.x = fvividLight(s.x,d.x);
	c.y = fvividLight(s.y,d.y);
	c.z = fvividLight(s.z,d.z);
	return c;
}

fn linearLight( s : vec3f, d : vec3f ) -> vec3f
{
	return 2.0 * s + d - 1.0;
}

fn fpinLight( s : f32, d : f32 ) -> f32
{
	return select(select(d, 2.0 * s,  s < 0.5 * d), 2.0 * s - 1.0, 2.0 * s - 1.0 > d);
}

fn pinLight( s : vec3f, d : vec3f ) -> vec3f
{
	var c : vec3f;
	c.x = fpinLight(s.x,d.x);
	c.y = fpinLight(s.y,d.y);
	c.z = fpinLight(s.z,d.z);
	return c;
}

fn hardMix( s : vec3f, d : vec3f ) -> vec3f
{
	return floor(s + d);
}

fn difference( s : vec3f, d : vec3f ) -> vec3f
{
	return abs(d - s);
}

fn exclusion( s : vec3f, d : vec3f ) -> vec3f
{
	return s + d - 2.0 * s * d;
}

fn subtract( s : vec3f, d : vec3f ) -> vec3f
{
	return s - d;
}

fn divide( s : vec3f, d : vec3f ) -> vec3f
{
	return s / d;
}