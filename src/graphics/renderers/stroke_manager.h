#pragma once

#include "graphics/edit.h"

#include <vector>

#define STROKE_HISTORY_MAX_SIZE 1000u

class SculptInstance;

struct sStrokeInfluence {
    uint32_t stroke_count = 0u;
    uint32_t pad_1 = UINT32_MAX; // aligment issues when using vec3
    uint32_t pad_0 = 0u;
    uint32_t pad_2 = UINT32_MAX;
    glm::vec4 pad1;
    glm::vec4 pad2;
    glm::vec4 pad3;
    Stroke strokes[STROKE_HISTORY_MAX_SIZE];
    glm::vec4 padd; // TODO(Juan): HACK esto no deveria ser necesario
};

struct sToComputeStrokeData {
    Stroke in_frame_stroke = {};
    sStrokeInfluence in_frame_influence;
    AABB in_frame_stroke_aabb;
};

struct StrokeManager {
    Stroke in_frame_stroke = {
        .stroke_id = 0u
    };
    Stroke current_stroke = {};

    std::vector<Stroke>* history = nullptr;
    std::vector<Stroke> redo_history;

    uint32_t pop_count_from_history = 0u;
    uint32_t redo_pop_count_from_history = 0u;

    glm::vec3 brick_world_size = {};

    StrokeParameters dirty_stroke_params;
    uint32_t dirty_stroke_increment = 0u;
    bool must_change_stroke = false;

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

    void undo(sToComputeStrokeData& result);
    void redo(sToComputeStrokeData& result);
    void add(std::vector<Edit> new_edits, sToComputeStrokeData &result);

    void update();

    AABB compute_grid_aligned_AABB(const AABB &base, const glm::vec3 &brick_world_size);
    void compute_history_intersection(sStrokeInfluence &influence, AABB& operation_aabb, uint32_t history_max_index);
};
