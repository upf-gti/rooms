#pragma once

#include "rooms_includes.h"

#include "graphics/edit.h"

#include <vector>

class SculptInstance;


struct StrokeManager {
    Stroke in_frame_stroke = {
        .stroke_id = 0u,
        .operation = OP_SMOOTH_UNION,
    };
    Stroke current_stroke = {
        .operation = OP_SMOOTH_UNION
    };

    std::vector<Stroke>* history = nullptr;
    std::vector<Stroke> redo_history;

    uint32_t pop_count_from_history = 0u;
    uint32_t redo_pop_count_from_history = 0u;

    glm::vec3 brick_world_size = {};

    StrokeParameters dirty_stroke_params;
    uint32_t dirty_stroke_increment = 0u;
    bool must_change_stroke = false;

    sStrokeInfluence result_to_compute;
    uint32_t edit_list_count = 0u;
    std::vector<Edit> edit_list;

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

    sStrokeInfluence* undo();
    sStrokeInfluence* redo();
    sStrokeInfluence* add(std::vector<Edit> new_edits);
    sStrokeInfluence* new_history_add(std::vector<Stroke>* history);

    void update();

    uint32_t divide_AABB_on_max_eval_size(const AABB& base, AABB divided_bases[8]);
    AABB compute_grid_aligned_AABB(const AABB &base, const glm::vec3 &brick_world_size);
    void compute_history_intersection(sStrokeInfluence &influence, AABB& operation_aabb, uint32_t history_max_index);
};
