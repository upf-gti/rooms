#pragma once

#include "includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#define PREVIEW_EDITS_MAX 128
#define EDITS_MAX 512
#define SDF_RESOLUTION 512

class RaymarchingRenderer {

    Uniform         u_sampler;

    // Compute
    Pipeline        initialize_sdf_pipeline;
    Shader*         initialize_sdf_shader = nullptr;
    WGPUBindGroup   initialize_sdf_bind_group = nullptr;

    Pipeline        compute_raymarching_pipeline;
    Shader*         compute_raymarching_shader = nullptr;
    WGPUBindGroup   compute_raymarching_textures_bind_group = nullptr;
    WGPUBindGroup   compute_raymarching_data_bind_group = nullptr;

    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Pipeline        compute_merge_pipeline;
    Shader*         compute_merge_shader = nullptr;
    WGPUBindGroup   compute_merge_bind_group = nullptr;

    Texture         sdf_texture;
    Texture         sdf_copy_read_texture;
    Uniform         compute_texture_sdf_storage_uniform;
    //Uniform         compute_texture_sdf_copy_storage_uniform;

    // Octree creation
    Pipeline        compute_octree_evaluate_pipeline;
    Pipeline        compute_octree_increment_level_pipeline;
    Pipeline        compute_octree_write_to_texture_pipeline;
    Shader*         compute_octree_evaluate_shader = nullptr;
    Shader*         compute_octree_increment_level_shader = nullptr;
    Shader*         compute_octree_write_to_texture_shader = nullptr;
    WGPUBindGroup   compute_octree_evaluate_bind_group = nullptr;
    WGPUBindGroup   compute_octree_increment_level_bind_group = nullptr;
    WGPUBindGroup   compute_octree_write_to_texture_bind_group = nullptr;
    WGPUBindGroup   compute_octant_usage_bind_groups[2] = {};
    Uniform         octree_uniform;
    Uniform         octant_usage_uniform[4];
    uint8_t         octree_depth = 0;
    Uniform         octree_indirect_buffer;
    Uniform         octree_atomic_counter;
    Uniform         octree_current_level;
    Uniform         octree_proxy_instance_buffer;
    Uniform         octree_proxy_indirect_buffer;

    Uniform         camera_uniform;

    Uniform         compute_buffer_data_uniform;
    Uniform         compute_preview_edit_uniform;
    Uniform         compute_texture_left_eye_uniform;
    Uniform         compute_texture_right_eye_uniform;

    Uniform         compute_merge_data_uniform;
    Uniform         compute_edits_array_uniform;

    // Data needed for XR raymarching
    struct sComputeData {
        glm::mat4x4 view_projection_left_eye;
        glm::mat4x4 view_projection_right_eye;

        glm::mat4x4 inv_view_projection_left_eye;
        glm::mat4x4 inv_view_projection_right_eye;

        glm::vec3 left_eye_pos;
        float render_height = 0.0f;
        glm::vec3 right_eye_pos;
        float render_width = 0.0f;

        float time          = 0.0f;
        float camera_near   = 0.0f;
        float camera_far    = 0.0f;
        float dummy0        = 0.0f;

        glm::vec3 sculpt_start_position = {0.f, 0.f, 0.f};
        float dummy1 = 0.0f;

        glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };

    } compute_raymarching_data;

    // Data needed for sdf merging
    struct sMergeData {
        glm::uvec3 edits_aabb_start = {};
        uint32_t edits_to_process = 0;
        glm::vec3  sculpt_start_position = { 0.f, 0.f, 0.f };
        uint32_t max_octree_depth = 0;
        glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    } compute_merge_data;

    struct sOctreeNode {
        uint32_t tile_pointer = 0;
    };

    std::vector<Edit> scene_edits;
    Edit* edits = nullptr;

    // Preview edits
    struct sPreviewEditsData {
        glm::vec3 aabb_center;
        float padding;
        glm::vec3 aabb_size;
        uint32_t preview_edits_count = 0u;
        Edit preview_edits[PREVIEW_EDITS_MAX];
    } preview_edit_data;

    // Timestepping counters
    float updated_time = 0.0f;

    void compute_initialize_sdf();

    void init_compute_raymarching_pipeline();
    void init_initialize_sdf_pipeline();
    void init_compute_merge_pipeline();
    void init_compute_octree_pipeline();
    void init_raymarching_proxy_pipeline();

public:

    RaymarchingRenderer();

    int initialize(bool use_mirror_screen);
    void clean();

    void update(float delta_time);
    void render();

    void compute_octree();
    void compute_merge();
    void compute_raymarching();
    void render_raymarching_proxy();

    void init_compute_raymarching_textures();

    void set_sculpt_start_position(const glm::vec3& position);
    void set_render_size(float width, float height);
    void set_left_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection);
    void set_right_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection);
    void set_near_far(float z_near, float z_far);

    /*
    *   Edits
    */

    void push_edit(Edit edit) {
        edits[compute_merge_data.edits_to_process++] = edit;
        scene_edits.push_back(edit);
    };

    void push_edit_list(std::vector<Edit> &new_edits) {
        for (Edit &edit : new_edits) {
            edits[compute_merge_data.edits_to_process++] = edit;
            scene_edits.push_back(edit);
        }
    };

    void add_preview_edit(const Edit& edit);
    void set_sculpt_rotation(const glm::quat& rotation);

    const std::vector<Edit>& get_scene_edits() { return scene_edits; }
    const glm::vec3& get_sculpt_start_position() { return compute_merge_data.sculpt_start_position; }
};
