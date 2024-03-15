#include math.wgsl

@group(0) @binding(0) var input_panorama_texture: texture_2d<f32>;
@group(0) @binding(1) var output_cubemap_texture: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var texture_sampler : sampler;

// From https://github.com/eliemichel/LearnWebGPU-Code/blob/step220/resources/compute-shader.wgsl

struct CubeMapUVL {
    uv: vec2f,
    layer: u32,
}

fn directionFromCubeMapUVL(uvl: CubeMapUVL) -> vec3f {
    let s = uvl.uv.x;
    let t = uvl.uv.y;
    let abs_ma = 1.0;
    let sc = 2.0 * s - 1.0;
    let tc = 2.0 * t - 1.0;
    var direction = vec3f(0.0);
    switch (uvl.layer) {
        case 0u {
            return vec3f(1.0, -tc, -sc);
        }
        case 1u {
            return vec3f(-1.0, -tc, sc);
        }
        case 2u {
            return vec3f(sc, 1.0, tc);
        }
        case 3u {
            return vec3f(sc, -1.0, -tc);
        }
        case 4u {
            return vec3f(sc, -tc, 1.0);
        }
        case 5u {
            return vec3f(-sc, -tc, -1.0);
        }
        default {
            return vec3f(0.0); // should not happen
        }
    }
}

fn textureGatherWeights_2df(t: texture_2d<f32>, uv: vec2f) -> vec2f {
    let dim = textureDimensions(t).xy;
    let scaled_uv = uv * vec2f(dim);
    // This is not accurate, see see https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
    // but bottom line is:
    //   "Unfortunately, if we need this to work, there seems to be no option but to check
    //    which hardware you are running on and apply the offset or not accordingly."
    return fract(scaled_uv - 0.5);
}

@compute @workgroup_size(4, 4, 6)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_dimensions = textureDimensions(output_cubemap_texture).xy;
    let layer = id.z;

    let uv = vec2f(id.xy) / vec2f(output_dimensions);

    let direction = directionFromCubeMapUVL(CubeMapUVL(uv, layer));

    let theta = acos(direction.z / length(direction));
    let phi = atan2(direction.y, direction.x);
    let latlong_uv = vec2f(phi / (2.0 * M_PI) + 0.5, theta / M_PI);

    let samples = array<vec4f, 4>(
        textureGather(0, input_panorama_texture, texture_sampler, latlong_uv),
        textureGather(1, input_panorama_texture, texture_sampler, latlong_uv),
        textureGather(2, input_panorama_texture, texture_sampler, latlong_uv),
        textureGather(3, input_panorama_texture, texture_sampler, latlong_uv),
    );

    let w = textureGatherWeights_2df(input_panorama_texture, latlong_uv);
    // TODO: could be represented as a matrix/vector product
    let color = vec4f(
        mix(mix(samples[0].w, samples[0].z, w.x), mix(samples[0].x, samples[0].y, w.x), w.y),
        mix(mix(samples[1].w, samples[1].z, w.x), mix(samples[1].x, samples[1].y, w.x), w.y),
        mix(mix(samples[2].w, samples[2].z, w.x), mix(samples[2].x, samples[2].y, w.x), w.y),
        mix(mix(samples[3].w, samples[3].z, w.x), mix(samples[3].x, samples[3].y, w.x), w.y),
    );

    textureStore(output_cubemap_texture, id.xy, layer, color);
}