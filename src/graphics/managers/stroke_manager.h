#pragma once

#include "rooms_includes.h"

#include "graphics/edit.h"

#include <vector>

class SculptNode;

enum eEvaluationFlags : uint32_t {
    UNDO_EVAL_FLAG = 1u,
    PREVIEW_EVAL_FLAG = 1u << 1u,
    PAINT_UNDO_EVAL_FLAG = 1u << 2u,
};

struct StrokeManager {
    uint32_t current_top_stroke_id = 0u;
    std::vector<Stroke>* history = nullptr;
    std::vector<Stroke> redo_history;

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

    void set_current_sculpt(SculptNode* sculpt_instance);

    void set_brick_world_size(const glm::vec3& new_brick_world_size) {
        brick_world_size = new_brick_world_size;
    }

    inline void request_new_stroke(const StrokeParameters& params, const uint32_t index_increment = 1u) {
        must_change_stroke = true;
        dirty_stroke_increment = index_increment;
        dirty_stroke_params = params;
    }

    void change_stroke_params(const StrokeParameters& params, const uint32_t index_increment = 1u);
    void change_stroke_params(const uint32_t index_increment = 1u);

    sStrokeInfluence* undo();
    sStrokeInfluence* redo();
    sStrokeInfluence* add(std::vector<Edit> new_edits);
    sStrokeInfluence* new_history_add(std::vector<Stroke>* history);

    uint32_t divide_AABB_on_max_eval_size(const AABB& base, AABB divided_bases[8]);
    AABB compute_grid_aligned_AABB(const AABB &base, const glm::vec3 &brick_world_size);
    void compute_history_intersection(sStrokeInfluence &influence, AABB& operation_aabb, uint32_t history_max_index);
};
