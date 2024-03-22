#include pbr_functions.wgsl

struct Uniforms {
    current_mip_level: u32,
    mip_level_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var input_cubemap_texture: texture_cube<f32>;
@group(0) @binding(1) var output_cubemap_texture: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var texture_sampler : sampler;
#dynamic @group(0) @binding(3) var<uniform> uniforms: Uniforms;

struct CubeMapUVL {
    uv: vec2f,
    layer: u32,
}

const SAMPLE_COUNT = 1024u;

// vec4 importanceSample = getImportanceSample(i, N, roughness);
// vec3 H = importanceSample.xyz;
// // float pdf = importanceSample.w;
// vec3 L = normalize(reflect(-V, H));

// float NdotL = saturate(L.z);
// float NdotH = saturate(H.z);
// float VdotH = saturate(dot(V, H));


// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/shaders/ibl_filtering.frag#L257
// Mipmap Filtered Samples (GPU Gems 3, 20.4)
// https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
// https://cgg.mff.cuni.cz/~jaroslav/papers/2007-sketch-fis/Final_sap_0073.pdf
fn compute_lod(pdf : f32, texture_width : f32) -> f32
{
    // https://cgg.mff.cuni.cz/~jaroslav/papers/2007-sketch-fis/Final_sap_0073.pdf
    return 0.5 * log2( 6.0 * texture_width * texture_width / (f32(SAMPLE_COUNT) * pdf));
}

fn directionFromCubeMapUVL(uvl: CubeMapUVL) -> vec3f {

    let uvx = 2.0 * uvl.uv.x - 1.0;
    let uvy = 2.0 * uvl.uv.y - 1.0;
    switch (uvl.layer) {
        case 0u {
            return vec3f(1.0, uvy, -uvx);
        }
        case 1u {
            return vec3f(-1.0, uvy, uvx);
        }
        case 2u {
            return vec3f(uvx, -1.0, uvy);
        }
        case 3u {
            return vec3f(uvx, 1.0, -uvy);
        }
        case 4u {
            return vec3f(uvx,  uvy,  1.0);
        }
        case 5u {
            return vec3f(-uvx, uvy, -1.0);
        }
        default {
            return vec3f(0.0); // should not happen
        }
    }
}

fn cubeMapUVLFromDirection(direction: vec3f) -> CubeMapUVL {
    let abs_direction = abs(direction);
    var major_axis_idx = 0u;
    //  Using the notations of https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf page 253
    // as suggested here: https://stackoverflow.com/questions/55558241/opengl-cubemap-face-order-sampling-issue
    var ma = 0.0;
    var sc = 0.0;
    var tc = 0.0;
    if (abs_direction.x > abs_direction.y && abs_direction.x > abs_direction.z) {
        major_axis_idx = 0u;
        ma = direction.x;
        if (ma >= 0) {
            sc = -direction.z;
        } else {
            sc = direction.z;
        }
        tc = direction.y;
    } else if (abs_direction.y > abs_direction.x && abs_direction.y > abs_direction.z) {
        major_axis_idx = 1u;
        ma = -direction.y;
        sc = direction.x;
        if (ma <= 0) {
            tc = direction.z;
        } else {
            tc = -direction.z;
        }
    } else {
        major_axis_idx = 2u;
        ma = direction.z;
        if (ma >= 0) {
            sc = direction.x;
        } else {
            sc = -direction.x;
        }
        tc = direction.y;
    }
    var sign_offset = 0u;
    if (ma < 0) {
        sign_offset = 1u;
    }
    let s = 0.5 * (sc / abs(ma) + 1.0);
    let t = 0.5 * (tc / abs(ma) + 1.0);
    return CubeMapUVL(
        vec2f(s, t),
        2 * major_axis_idx + sign_offset,
    );
}

fn makeLocalFrame(N: vec3f) -> mat3x3f {
    let Z = N;
    var up = vec3f(0.0, 0.0, 1.0);
    if (abs(N.z) > abs(N.x) && abs(N.z) > abs(N.y)) {
        up = vec3f(0.0, 1.0, 0.0);
    }
    let X = normalize(cross(up, Z));
    let Y = normalize(cross(Z, X));
    return mat3x3f(X, Y, Z);
}

fn textureGatherWeights_cubef(t: texture_cube<f32>, direction: vec3f) -> vec2f {
    // major axis direction
    let cubemap_uvl = cubeMapUVLFromDirection(direction);
    let dim = textureDimensions(t).xy;
    var uv = cubemap_uvl.uv;

    // Empirical fix...
    if (cubemap_uvl.layer == 4u || cubemap_uvl.layer == 5u) {
        uv.x = 1.0 - uv.x;
    }

    let scaled_uv = uv * vec2f(dim);
    // This is not accurate, see see https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
    // but bottom line is:
    //   "Unfortunately, if we need this to work, there seems to be no option but to check
    //    which hardware you are running on and apply the offset or not accordingly."
    return fract(scaled_uv - 0.5);
}

fn sampleCubeMap(cubemapTexture: texture_cube<f32>, direction: vec3f) -> vec4f {
    let samples = array<vec4f, 4>(
        textureGather(0, cubemapTexture, texture_sampler, direction),
        textureGather(1, cubemapTexture, texture_sampler, direction),
        textureGather(2, cubemapTexture, texture_sampler, direction),
        textureGather(3, cubemapTexture, texture_sampler, direction),
    );

    let w = textureGatherWeights_cubef(cubemapTexture, direction);
    
    return vec4f(
        mix(mix(samples[0].w, samples[0].z, w.x), mix(samples[0].x, samples[0].y, w.x), w.y),
        mix(mix(samples[1].w, samples[1].z, w.x), mix(samples[1].x, samples[1].y, w.x), w.y),
        mix(mix(samples[2].w, samples[2].z, w.x), mix(samples[2].x, samples[2].y, w.x), w.y),
        mix(mix(samples[3].w, samples[3].z, w.x), mix(samples[3].x, samples[3].y, w.x), w.y),
    );
}

// https://github.com/eliemichel/LearnWebGPU-Code/blob/step222/resources/compute-shader.wgsl#L448
// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/shaders/ibl_filtering.frag#L283
@compute @workgroup_size(4, 4, 6)
fn compute(@builtin(global_invocation_id) id: vec3u) 
{
    let layer = id.z;
    var color : vec3f = vec3f(0.0, 0.0, 0.0);
    var total_weight = 0.0;

    let roughness = f32(uniforms.current_mip_level) / f32(uniforms.mip_level_count - 1);

    let output_dimensions = textureDimensions(output_cubemap_texture).xy;
    var uv = vec2f(id.xy) / vec2f(output_dimensions);

    var N = normalize(directionFromCubeMapUVL(CubeMapUVL(uv, layer)));

    N.y *= -1;

    // let local_to_world = makeLocalFrame(N);

    for(var i : u32 = 0u; i < SAMPLE_COUNT; i = i + 1u)
    {
        let Xi : vec2f = Hammersley(i, SAMPLE_COUNT);

        let importance_sample : vec4f = importance_sample_GGX(Xi, N, roughness);
        let H : vec3f = normalize(importance_sample.xyz);
        let pdf : f32 = importance_sample.w;

        // var lod = compute_lod(pdf, f32(output_dimensions.x));

        // bias
        // lod += 1.0;

        let V : vec3f = N;
        let L : vec3f = normalize(reflect(-V, H));

        let NdotL : f32 = dot(N, L);

        if (NdotL > 0.0)
        {
            let radiance_ortho = sampleCubeMap(input_cubemap_texture, L).rgb;

            color += radiance_ortho * NdotL;
            total_weight += NdotL;
        }
    }

    if (total_weight != 0.0)
    {
        color /= total_weight;
    }
    else
    {
        color /= f32(SAMPLE_COUNT);
    }

    textureStore(output_cubemap_texture, id.xy, layer, vec4(color, 1.0));
}