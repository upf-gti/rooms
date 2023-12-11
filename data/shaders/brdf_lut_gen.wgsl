#include pbr_functions.wgsl

@group(0) @binding(0) var brdf_lut: texture_storage_2d<rg32float, write>;

const SAMPLE_COUNT = 1024u;

fn integrate_BRDF(n_dot_v : f32, roughness : f32) -> vec2f
{
    var V : vec3f;
    V.x = sqrt(1.0 - n_dot_v * n_dot_v);
    V.y = 0.0;
    V.z = n_dot_v;

    let roughness_clamp : f32 = max(roughness, 0.0001);

    var A : f32 = 0.0;
    var B : f32 = 0.0;

    let N : vec3f = vec3f(0.0, 0.0, 1.0);

    for(var i : u32 = 0u; i < SAMPLE_COUNT; i = i + 1)
    {
        let Xi : vec2f = Hammersley(i, SAMPLE_COUNT);
        let H : vec3f = importance_sample_GGX(Xi, N, roughness_clamp);
        let L : vec3f = 2.0 * dot(V, H) * H - V;

        let NdotL : f32 = max(L.z, 0.0);
        let NdotH : f32 = max(H.z, 0.0);
        let VdotH : f32 = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            let G : f32 = GDFG(n_dot_v, NdotL, roughness_clamp);
            let G_Vis : f32 = (G * VdotH) / (NdotH);
            let Fc : f32 = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);

    return vec2f(A, B);
}

@compute @workgroup_size(32, 32, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) 
{
    textureStore(brdf_lut, id.xy, vec4f(integrate_BRDF(f32(id.x) / 512.0, f32(id.y) / 512.0), 0.0, 0.0));
}
