#pragma once

#include "includes.h"
#include "graphics/renderer.h"
#include <vector>
#include "graphics/edit.h"

#include "graphics/texture.h"

#include "tools/sculpt/tool.h"

#define EDITS_MAX 1024
#define SDF_RESOLUTION 512

class RaymarchingRenderer {

    Uniform                 u_render_texture_left_eye;
    Uniform                 u_render_texture_right_eye;
    Uniform                 u_sampler;

    // Compute
    Pipeline                initialize_sdf_pipeline;
    Shader*                 initialize_sdf_shader = nullptr;
    WGPUBindGroup           initialize_sdf_bind_group = nullptr;

    Pipeline                compute_raymarching_pipeline;
    Shader*                 compute_raymarching_shader = nullptr;
    WGPUBindGroup           compute_raymarching_textures_bind_group = nullptr;
    WGPUBindGroup           compute_raymarching_data_bind_group = nullptr;

    Pipeline                compute_merge_pipeline;
    Shader*                 compute_merge_shader = nullptr;
    WGPUBindGroup           compute_merge_bind_group = nullptr;

    Texture                 sdf_texture;
    Texture                 sdf_copy_read_texture;
    Uniform                 u_compute_texture_sdf_storage;
    Uniform                 u_compute_texture_sdf_copy_read_storage;

    Texture                 left_eye_texture;
    Texture                 left_eye_depth_texture;
    Texture                 right_eye_texture;
    Texture                 right_eye_depth_texture;

    // Octree creation
    Pipeline                compute_octree_flag_nodes_pipeline;
    Shader*                 compute_octree_flag_nodes_shader = nullptr;
    Uniform                 u_octree;

    Uniform                 u_camera;

    Uniform                 u_compute_buffer_data;
    Uniform                 u_compute_preview_edit;
    Uniform                 u_compute_texture_left_eye;
    Uniform                 u_compute_texture_right_eye;

    Uniform                 u_compute_merge_data;
    Uniform                 u_compute_edits_array;

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

        glm::vec3 sculpt_start_position = {};
        float dummy1 = 0.0f;

        glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };;

    } compute_raymarching_data;

    // Data needed for sdf merging
    struct sMergeData {
        glm::uvec3 edits_aabb_start = {};
        uint32_t edits_to_process = 0;
        glm::vec3  sculpt_start_position = {};
        float dummy0;
        glm::quat sculpt_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    } compute_merge_data;

    struct sCameraData {
        glm::mat4x4 view_projection;
    } camera_data;

    Edit edits[EDITS_MAX];

    // Timestepping counters
    float updated_time = 0.0f;

    glm::vec3 clear_color;

    void render_eye_quad(WGPUTextureView swapchain_view, WGPUTextureView swapchain_depth, WGPUBindGroup bind_group);
    void render_screen();

#if defined(XR_SUPPORT)
    void render_xr();
#endif

    void compute_initialize_sdf(int sdf_texture_idx);

    void init_render_quad_pipeline();
    void init_render_quad_bind_groups();
    void init_render_mesh_pipelines();
    void init_compute_raymarching_pipeline();
    void init_compute_raymarching_textures();
    void init_initialize_sdf_pipeline();
    void init_compute_merge_pipeline();

    void init_compute_octree_pipeline();

#if defined(XR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    void render_mirror();
    void init_mirror_pipeline();
#endif

public:

    RaymarchingRenderer();

    int initialize(bool use_mirror_screen);
    void clean();

    void update(float delta_time);
    void render();

    void compute_merge();
    void compute_raymarching();

    void set_sculpt_start_position(const glm::vec3& position) {
        compute_merge_data.sculpt_start_position = position;
        compute_raymarching_data.sculpt_start_position = position;
    }

    void set_left_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection);
    void set_right_eye(const glm::vec3& eye_pos, const glm::mat4x4& view_projection);
    void set_near_far(float z_near, float z_far);

    /*
    *   Edits
    */

    void push_edit(Edit edit) {
        edits[compute_merge_data.edits_to_process++] = edit;
    };

    void push_edit_list(std::vector<Edit> &edits) {
        for (Edit edit : edits) {
            edits[compute_merge_data.edits_to_process++] = edit;
        }
    };

    void set_preview_edit(const Edit& edit);
    void set_sculpt_rotation(const glm::quat& rotation);

};
