#pragma once

#include "includes.h"
#include "rooms_includes.h"

#include "graphics/pipeline.h"
#include "graphics/edit.h"
#include "graphics/texture.h"

#include "framework/nodes/sculpt_node.h"
#include "framework/math/aabb.h"

#include "graphics/managers/stroke_manager.h"

#include <list>


/*
    TODO:
        - Make dynamic the instance data buffers
        - Preview
*/

class MeshInstance3D;

enum eSculptInstanceContextFlags : uint32_t {
    SCULPT_NOT_SELECTED = 0u,
    SCULPT_IS_OUT_OF_FOCUS = 1 << 0,
    SCULPT_IS_HOVERED = 1 << 1,
    SCULPT_IS_SELECTED = 1 << 2,
    SCULPT_IN_SCULPT_EDITOR = 1 << 3,
    SCULPT_IN_SCENE_EDITOR = 1 << 4,
    SCULPT_HOVER_CHECK_SIBLINGS = 1 << 5,
};

#define MAX_INSTANCES_PER_SCULPT 20u

class RaymarchingRenderer {

    enum eEvaluatorOperationFlags : uint32_t {
        CLEAN_BEFORE_EVAL = 0x0001u,
        EVALUATE_PREVIEW_STROKE = 0x0002u
    };

    // Render pipelines
    Pipeline        render_proxy_geometry_pipeline;
    Shader*         render_proxy_shader = nullptr;
    WGPUBindGroup   render_proxy_geometry_bind_group = nullptr;

    Pipeline        render_preview_proxy_geometry_pipeline;
    Shader*         render_preview_proxy_shader = nullptr;

    WGPUBindGroup   sculpt_data_bind_preview_group = nullptr;

    Uniform*        camera_uniform;
    WGPUBindGroup   render_camera_bind_group = nullptr;

    MeshInstance3D* cube_mesh = nullptr;

    bool            render_preview = false;

    uint32_t preview_edit_array_length = 0u;
    struct sPreviewStroke {
        uint32_t current_sculpt_idx;
        uint32_t dummy0;
        uint32_t dummy1;
        uint32_t dummy2;
        sGPUStroke stroke;
        std::vector<Edit> edit_list;

        AABB get_AABB() const;
    } preview_stroke;

    // Timestepping counters
    float updated_time = 0.0f;

    void init_raymarching_proxy_pipeline();

    void render_raymarching_proxy(WGPURenderPassEncoder render_pass, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);

    void render_preview_raymarching_proxy(WGPURenderPassEncoder render_pass, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);


    // DEBUG
    MeshInstance3D* AABB_mesh = nullptr;

public:

    RaymarchingRenderer();

    int initialize();

    void clean();

    void render(WGPURenderPassEncoder render_pass, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0u);

    inline void set_preview_render(const bool need_to_render_preview) {
        render_preview = need_to_render_preview;
    }

    /*
    *   Edits
    */

    inline void add_preview_edit(const Edit& edit) {
        if (preview_stroke.stroke.edit_count == preview_stroke.edit_list.size()) {
            preview_stroke.edit_list.resize(preview_stroke.edit_list.size() + PREVIEW_EDIT_LIST_INCREMENT);
        }
        preview_stroke.edit_list[preview_stroke.stroke.edit_count++] = edit;
    }
};
