const PI = 3.14159265359;

// Gloss = 1 - rough*rough
fn Specular_F_Roughness(specular_color : vec3f, gloss : f32, h : vec3f, v : vec3f) -> vec3f
{
  // Sclick using roughness to attenuate fresnel.
  return (specular_color + (max(vec3f(gloss), specular_color) - specular_color) * pow((1.0 - saturate(dot(v, h))), 5.0));
}

fn Diffuse(albedo : vec3f) -> vec3f
{
    return albedo / PI;
}

fn DistributionGGX(N : vec3f, H : vec3f, roughness : f32) -> f32
{
    var a : f32 = roughness * roughness;
    var a2 : f32 = a * a;
    var NdotH : f32 = max(dot(N, H), 0.0);
    var NdotH2 : f32 = NdotH * NdotH;

    var nom : f32   = a2;
    var denom : f32 = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

fn DistributionGGX2(N : vec3f, H : vec3f, roughness : f32) -> f32
{
    var NoH : f32 = clamp(dot(N, H), 0.0, 1.0);
    var a : f32 = NoH * roughness;
    var k : f32 = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}

fn GeometrySchlickGGX(NdotV : f32, roughness : f32) -> f32
{
    var r : f32 = (roughness + 1.0);
    var k : f32 = (r * r) / 8.0;

    var nom : f32   = NdotV;
    var denom : f32 = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn V_SmithGGXCorrelated(N : vec3f, V : vec3f, L : vec3f, roughness : f32) -> f32
{
    var NoV : f32 = abs(dot(N, V)) + 1e-5;
    var NoL : f32 = clamp(dot(N, L), 0.0, 1.0);
    var a2 : f32 = roughness * roughness;
    var GGXV : f32 = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    var GGXL : f32 = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

fn GeometrySmith(N : vec3f, V : vec3f, L : vec3f, roughness : f32) -> f32
{
    var NdotV : f32 = max(dot(N, V), 0.0);
    var NdotL : f32 = max(dot(N, L), 0.0);
    var ggx2 : f32 = GeometrySchlickGGX(NdotV, roughness);
    var ggx1 : f32 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

fn GDFG(NoV : f32, NoL : f32, a : f32) -> f32
{
    let a2 : f32 = a * a;
    let GGXL : f32 = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
    let GGXV : f32 = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    return (2 * NoL) / (GGXV + GGXL);
}

fn importance_sample_GGX(Xi : vec2f, N : vec3f, roughness : f32) -> vec3f
{
    let a : f32 = roughness * roughness;
	
    let phi : f32 = 2.0 * PI * Xi.x;
    let cos_theta : f32 = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    let sin_theta : f32 = sqrt(1.0 - cos_theta * cos_theta);
	
    // from spherical coordinates to cartesian coordinates
    let H : vec3f = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    
    // from tangent-space vector to world-space sample vector
    let up : vec3f        = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent : vec3f   = normalize(cross(up, N));
    let bitangent : vec3f = cross(N, tangent);
	
    let sampleVec : vec3f = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

fn RadicalInverse_VdC(bits_in : u32) -> f32
{
    var bits : u32 = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn Hammersley(i : u32, N : u32) -> vec2f
{
    return vec2f(f32(i)/ f32(N), RadicalInverse_VdC(i));
}  

fn fresnelSchlick(cos_theta : f32, F0 : vec3f) -> vec3f
{
    return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn Fresnel_Schlick( specular_color : vec3f, h : vec3f, v : vec3f) -> vec3f
{
    return (specular_color + (1.0 - specular_color) * pow((1.0 - saturate(dot(v, h))), 5.0));
}

fn FresnelSchlickRoughness(cos_theta : f32, F0 : vec3f, roughness : f32) -> vec3f
{
    return F0 + (max(vec3f(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

//Javi Agenjo Snipet for Bump Mapping

fn cotangent_frame( N : vec3f, p : vec3f, uv : vec2f ) -> mat3x3f
{
    // get edge vectors of the pixel triangle
    var dp1 : vec3f = dpdx( p );
    var dp2 : vec3f = dpdy( p );
    var duv1 : vec2f = dpdx( uv );
    var duv2 : vec2f = dpdy( uv );

    // solve the linear system
    var dp2perp : vec3f = cross( dp2, N );
    var dp1perp : vec3f = cross( N, dp1 );
    var T : vec3f = dp2perp * duv1.x + dp1perp * duv2.x;
    var B : vec3f = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    var invmax : f32 = inverseSqrt( max( dot(T,T), dot(B,B) ) );
    return mat3x3( T * invmax, B * invmax, N );
}

fn perturb_normal( N : vec3f, V : vec3f, texcoord : vec2f, normal_color : vec3f ) -> vec3f
{
    // assume N, the interpolated vertex normal and
    // V, the view vector (vertex to eye)
    var normal_pixel = normal_color * (255.0/127.0) - vec3f(128.0/127.0);
    var TBN : mat3x3f = cotangent_frame(N, V, texcoord);
    return normalize(TBN * normal_pixel);
}
