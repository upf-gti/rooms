#include ../sdf_functions.wgsl

struct OctreeNode {
    tile_pointer : u32
}

struct Octree {
    data : array<OctreeNode>
};

@group(0) @binding(0) var<uniform> edits : Edits;
@group(0) @binding(1) var<storage> octree : Octree;
@group(0) @binding(2) var<storage, read_write> indirect_buffer : vec3u;
@group(0) @binding(3) var write_sdf: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> atomic_counter : atomic<u32>;

@group(1) @binding(0) var<storage, read> octant_usage_read : array<u32>;
@group(1) @binding(1) var<storage> octant_usage_write : array<u32>;

@compute @workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {

}
