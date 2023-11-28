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

// Lights

struct LitMaterial
{
  pos : vec3f,
  normal : vec3f,
  albedo : vec3f,
  emissive : vec3f,
  diffuse_color : vec3f,
  specular_color : vec3f,
  metallic : f32,
  roughness : f32,
  ao : f32,
  view_dir : vec3f,
  reflected_dir : vec3f
};

fn get_direct_light( m : LitMaterial, shadow_factor : vec3f, attenuation : f32) -> vec3f
{
    var N : vec3f = normalize(m.normal);
    var V : vec3f = normalize(m.view_dir);

    var F0 : vec3f = m.specular_color;

    // hardcoded!!
    let light_position = vec3f(0.2, 0.5, 1.0);
    let light_color = vec3f(1.0);
    let light_intensity = 5.0;

    // calculate light radiance
    var L : vec3f = normalize(light_position - m.pos);
    var H : vec3f = normalize(V + L);
    var radiance : vec3f = light_color * light_intensity * attenuation * shadow_factor;

    // Cook-Torrance BRDF
    var NDF : f32 = DistributionGGX2(N, H, m.roughness);
    var G : f32 = GeometrySmith(N, V, L, m.roughness);
    var F : vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);

    var numerator : vec3f  = NDF * G * F;
    var denominator : f32 = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    var specular : vec3f = numerator / denominator;

    var k_s : vec3f = F;
    var k_d : vec3f = vec3f(1.0) - k_s;

    var NdotL : f32 = max(dot(N, L), 0.0);

    var final_color : vec3f = ((k_d * Diffuse(m.diffuse_color)) + specular) * radiance * NdotL;

    return final_color;
}

// TOnemapping

// Uncharted 2 tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
fn tonemap_uncharted2_imp( color : vec3f ) -> vec3f
{
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

fn tonemap_uncharted( c : vec3f ) -> vec3f
{
    let W = 11.2;
    let color = tonemap_uncharted2_imp(c * 2.0);
    let whiteScale = 1.0 / tonemap_uncharted2_imp( vec3f(W) );
    return color * whiteScale;//LINEARtoSRGB(color * whiteScale);
}