#pragma once

#include "graphics/edit.h"

#include <vector>

#define EDIT_BUFFER_INITIAL_SIZE 512u
#define EDIT_BUFFER_INCREASE 512u
#define STROKE_CONTEXT_INTIAL_SIZE 100u
#define STROKE_CONTEXT_INCREASE 100u
#define AREA_MAX_EVALUATION_SIZE  (1.0f / 4.0f)

class SculptInstance;

struct sToUploadStroke {
    uint32_t        stroke_id = 0u;
    uint32_t        edit_count = 0u;
    sdPrimitive     primitive;
    sdOperation     operation;//4
    glm::vec4	    parameters = { 0.f, -1.f, 0.f, 0.f }; // 4
    glm::vec3	    aabb_min;// 4
    ColorBlendOp    color_blending_op = ColorBlendOp::COLOR_OP_REPLACE;
    glm::vec3	    aabb_max;
    uint32_t        edit_list_index = 0u;// 4
    // 48 bytes
    StrokeMaterial material;
};

struct sStrokeInfluence {
    uint32_t stroke_count = 0u;
    uint32_t pad_1 = UINT32_MAX; // aligment issues when using vec3
    uint32_t pad_0 = 0u;
    uint32_t pad_2 = UINT32_MAX;
    glm::vec3 eval_aabb_min;
    float pad1;
    glm::vec3 eval_aabb_max;
    float pad2;
    glm::vec4 pad3;
    std::vector<sToUploadStroke> strokes;
};

struct sToComputeStrokeData {
    sToUploadStroke in_frame_stroke = {};
    sStrokeInfluence in_frame_influence;
    AABB in_frame_stroke_aabb;

    inline void set_defaults() {
        in_frame_stroke.edit_count = 0u;
        in_frame_influence.stroke_count = 0u;
        in_frame_stroke_aabb = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
    }
};


struct StrokeManager {
    Stroke current_stroke = {};

    std::vector<Stroke>* history = nullptr;
    std::vector<Stroke> redo_history;

    uint32_t pop_count_from_history = 0u;
    uint32_t redo_pop_count_from_history = 0u;

    glm::vec3 brick_world_size = {};

    uint32_t edit_list_count = 0u;
    std::vector<Edit> edit_list;

    StrokeParameters dirty_stroke_params;
    uint32_t dirty_stroke_increment = 0u;
    bool must_change_stroke = false;

    sToComputeStrokeData result_to_compute;

    inline void add_edit_to_upload(const Edit& edit) {
        // Expand the edit to upload list by chunks
        if (edit_list.size() == edit_list_count) {
            edit_list.resize(edit_list.size() + EDIT_BUFFER_INCREASE);
        }

        edit_list[edit_list_count++] = edit;
    }

    void add_stroke_to_upload_list(sStrokeInfluence& influence, const Stroke& stroke);

    void init();

    void set_current_sculpt(SculptInstance* sculpt_instance);

    void set_brick_world_size(const glm::vec3& new_brick_world_size) {
        brick_world_size = new_brick_world_size;
    }

    inline void request_new_stroke(const StrokeParameters& params, const uint32_t index_increment = 1u) {
        must_change_stroke = true;
        dirty_stroke_increment = index_increment;
        dirty_stroke_params = params;
    }

    void change_stroke(const StrokeParameters& params, const uint32_t index_increment = 1u);
    void change_stroke(const uint32_t index_increment = 1u);

    sToComputeStrokeData* undo();
    sToComputeStrokeData* redo();
    sToComputeStrokeData* add(std::vector<Edit> new_edits);
    sToComputeStrokeData* new_history_add(std::vector<Stroke>* history);

    void update();

    uint32_t divide_AABB_on_max_eval_size(const AABB& base, AABB divided_bases[8]);
    AABB compute_grid_aligned_AABB(const AABB &base, const glm::vec3 &brick_world_size);
    void compute_history_intersection(sStrokeInfluence &influence, AABB& operation_aabb, uint32_t history_max_index);
};
