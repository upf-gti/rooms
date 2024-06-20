#include "stroke_manager.h"

#include "framework/nodes/sculpt_instance.h"

#include "spdlog/spdlog.h"

void StrokeManager::init() {
    edit_list.resize(EDIT_BUFFER_INITAL_SIZE);
}

void StrokeManager::add_stroke_to_upload_list(sStrokeInfluence& influence, const Stroke& stroke) {
    memcpy(&influence.strokes[influence.stroke_count], &stroke, sizeof(sToUploadStroke));
    influence.strokes[influence.stroke_count].edit_list_index = edit_list_count;

    AABB stroke_aabb = stroke.get_world_AABB();

    influence.strokes[influence.stroke_count].aabb_max = stroke_aabb.center + stroke_aabb.half_size;
    influence.strokes[influence.stroke_count].aabb_min = stroke_aabb.center - stroke_aabb.half_size;

    for (uint32_t j = 0u; j < stroke.edit_count; j++) {
        add_edit_to_upload(stroke.edits[j]);
    }

    influence.stroke_count++;
}

void StrokeManager::compute_history_intersection(sStrokeInfluence &influence, AABB& operation_aabb, uint32_t history_max_index) {
    Stroke intersection_stroke = {};
    // Get the strokes that are on the region of the undo
    for (uint32_t i = 0u; i < history_max_index; i++) {
        history->at(i).get_AABB_intersecting_stroke(operation_aabb, intersection_stroke);

        if (intersection_stroke.edit_count > 0u) {
            add_stroke_to_upload_list(influence, intersection_stroke);
        }
    }

    // Include the current stroke as context
    if (current_stroke.edit_count > 0u) {
        // Exclude the last one, since it is same edit
        current_stroke.get_AABB_intersecting_stroke(operation_aabb, intersection_stroke);

        if (intersection_stroke.edit_count > 0u) {
            add_stroke_to_upload_list(influence, intersection_stroke);
        }
    }
}

AABB StrokeManager::compute_grid_aligned_AABB(const AABB& base, const glm::vec3& brick_world_size) {

    glm::vec3 max_aabb = base.center + base.half_size;
    glm::vec3 min_aabb = base.center - base.half_size;

    // TODO: revisit this margin of 1 for the AABB size
    max_aabb = (glm::ceil(max_aabb / brick_world_size) + 1.0f) * brick_world_size;
    min_aabb = (glm::floor(min_aabb / brick_world_size) - 1.0f) * brick_world_size;

    const glm::vec3 half_size = (max_aabb - min_aabb) * 0.5f;

    return {
        .center = min_aabb + half_size,
        .half_size = half_size
    };
}


void StrokeManager::undo(sToComputeStrokeData& result) {
    if (history->size() <= 0u) {
        return;
    }
    uint32_t last_stroke_id = 0u;
    uint32_t united_stroke_idx = 0;
    uint32_t last_history_index = 0u;
    float max_smooth_margin = 0.0f;

    // Get the last stroke to undo, and compute the AABB
    for (united_stroke_idx = history->size(); united_stroke_idx > 0; --united_stroke_idx) {
        Stroke& prev = (*history)[united_stroke_idx - 1];

        // if stroke changes
        if (last_stroke_id != 0u && prev.stroke_id != last_stroke_id) {
            break;
        }

        result.in_frame_stroke_aabb = merge_aabbs(result.in_frame_stroke_aabb, prev.get_world_AABB());
        max_smooth_margin = glm::max(prev.parameters.w, max_smooth_margin);
        pop_count_from_history++;

        last_stroke_id = prev.stroke_id;
    }

    // In the case of first stroke, submit it as substraction to clear everything
    if (united_stroke_idx == 0) {
        last_history_index = 0;
    }
    else {
        //result.in_frame_stroke = history->data()[united_stroke_idx - 1];
        last_history_index = united_stroke_idx - 1;
    }

    result.in_frame_influence.eval_aabb_min = result.in_frame_stroke_aabb.center - result.in_frame_stroke_aabb.half_size;
    result.in_frame_influence.eval_aabb_max = result.in_frame_stroke_aabb.center + result.in_frame_stroke_aabb.half_size;

    // Fit the AABB to the eval grid
    result.in_frame_stroke_aabb.half_size += glm::vec3(max_smooth_margin);
    AABB culling_aabb = compute_grid_aligned_AABB(result.in_frame_stroke_aabb, brick_world_size);
    result.in_frame_stroke_aabb.half_size -= glm::vec3(max_smooth_margin);

    // Compute and fill intersection
    compute_history_intersection(result.in_frame_influence, culling_aabb, last_history_index);

    // In the case of first stroke, submit it as substraction to clear everything
    if (united_stroke_idx == 0) {
        Stroke prev = history->at(0);
        prev.operation = OP_SMOOTH_SUBSTRACTION;
        //result.in_frame_stroke = prev;

        add_stroke_to_upload_list(result.in_frame_influence, prev);
    } else {
        //result.in_frame_stroke = history->data()[united_stroke_idx - 1];
        add_stroke_to_upload_list(result.in_frame_influence, history->data()[united_stroke_idx - 1]);
    }
}


void StrokeManager::redo(sToComputeStrokeData& result) {
    if (redo_history.size() <= 0u) {
        return;
    }
    uint32_t strokes_to_redo_count = redo_history.size();
    uint32_t united_stroke_idx = redo_history.back().stroke_id;
    float max_smooth_margin = 0.0f;

    if (redo_history.size() == 1u) {
        max_smooth_margin = redo_history[0u].parameters.w;
        redo_pop_count_from_history = 1u;
        strokes_to_redo_count = 1u;
        result.in_frame_stroke_aabb = redo_history[0u].get_world_AABB();
    } else {
        // Get the last edit to redo, and compute the AABB
        for (; strokes_to_redo_count > 0u;) {
            Stroke& curr_stroke = redo_history[strokes_to_redo_count - 1u];

            if (united_stroke_idx != curr_stroke.stroke_id) {
                strokes_to_redo_count++;
                break;
            }

            max_smooth_margin = glm::max(max_smooth_margin, curr_stroke.parameters.w);
            redo_pop_count_from_history++;
            strokes_to_redo_count--;
            result.in_frame_stroke_aabb = merge_aabbs(result.in_frame_stroke_aabb, curr_stroke.get_world_AABB());
        }
    }
    
    spdlog::info("redo size: {}, to pop {}", redo_history.size(), redo_pop_count_from_history);

    add_stroke_to_upload_list(result.in_frame_influence, redo_history[strokes_to_redo_count - 1u]);

    result.in_frame_influence.eval_aabb_min = result.in_frame_stroke_aabb.center - result.in_frame_stroke_aabb.half_size;
    result.in_frame_influence.eval_aabb_max = result.in_frame_stroke_aabb.center + result.in_frame_stroke_aabb.half_size;

    // Fit the AABB to the eval grid
    result.in_frame_stroke_aabb.half_size += glm::vec3(max_smooth_margin);
    AABB culling_aabb = compute_grid_aligned_AABB(result.in_frame_stroke_aabb, brick_world_size);
    result.in_frame_stroke_aabb.half_size -= glm::vec3(max_smooth_margin);

    // Compute and fill intersection
    compute_history_intersection(result.in_frame_influence, culling_aabb, history->size());
}


void StrokeManager::add(std::vector<Edit> new_edits, sToComputeStrokeData& result) {
    // Add new edits to the current stroke and the in_frame_stroke
    for (uint8_t i = 0u; i < new_edits.size(); i++) {
        in_frame_stroke.edits[in_frame_stroke.edit_count++] = new_edits[i];
    }

    // Compute AABB for the incomming strokes
    result.in_frame_stroke_aabb = in_frame_stroke.get_world_AABB();
    AABB culling_aabb = compute_grid_aligned_AABB(result.in_frame_stroke_aabb, brick_world_size);
    result.in_frame_stroke_aabb.half_size -= glm::vec3(in_frame_stroke.parameters.w);

    // Compute and fill intersection
    compute_history_intersection(result.in_frame_influence, culling_aabb, history->size());

    add_stroke_to_upload_list(result.in_frame_influence, in_frame_stroke);

    for (uint8_t i = 0u; i < new_edits.size(); i++) {
        // if exceeds the maximun number of edits per stroke, store the current to the history
        // and add them to a new one, with the same ID
        if (current_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {
            history->push_back(current_stroke);
            current_stroke.edit_count = 0u;
        }

        current_stroke.edits[current_stroke.edit_count++] = new_edits[i];
    }

    redo_history.clear();

    result.in_frame_influence.eval_aabb_min = result.in_frame_stroke_aabb.center - result.in_frame_stroke_aabb.half_size;
    result.in_frame_influence.eval_aabb_max = result.in_frame_stroke_aabb.center + result.in_frame_stroke_aabb.half_size;
}

void StrokeManager::update() {
    // Remove undo strokes of history, and add them to the redo history
    for (uint32_t i = 0u; i < pop_count_from_history; i++) {
        redo_history.push_back(history->back());
        history->pop_back();
    }

    for (uint32_t i = 0u; i < redo_pop_count_from_history; i++) {
        history->push_back(redo_history.back());
        redo_history.pop_back();
    }

    if (must_change_stroke) {
        change_stroke(dirty_stroke_params, dirty_stroke_increment);
        must_change_stroke = false;
    }

    in_frame_stroke.edit_count = 0u;
    pop_count_from_history = 0u;
    redo_pop_count_from_history = 0u;
    edit_list_count = 0u;
}

void StrokeManager::change_stroke(const uint32_t index_increment) {
    if (current_stroke.edit_count > 0u) {
        // Add it to the history
        history->push_back(current_stroke);
    }

    current_stroke.edit_count = 0u;
    current_stroke.stroke_id = current_stroke.stroke_id + index_increment;
    spdlog::info("change stroke");
}

void StrokeManager::set_current_sculpt(SculptInstance* sculpt_instance)
{
    history = &sculpt_instance->get_stroke_history();

    if (!history->empty()) {
        current_stroke.stroke_id = history->back().stroke_id + 1;
        in_frame_stroke.stroke_id = history->back().stroke_id + 1;
    }
    else {
        current_stroke.stroke_id = 0;
        in_frame_stroke.stroke_id = 0;
    }

    redo_history.clear();

    pop_count_from_history = 0;
    redo_pop_count_from_history = 0;
    edit_list_count = 0u;
}

void StrokeManager::change_stroke(const StrokeParameters& params, const uint32_t index_increment) {
    if (current_stroke.edit_count > 0u) {
        // Add it to the history
        history->push_back(current_stroke);
    }

    current_stroke.edit_count = 0u;
    current_stroke.stroke_id = current_stroke.stroke_id + index_increment;
    current_stroke.primitive = params.get_primitive();
    current_stroke.operation = params.get_operation();
    current_stroke.color_blending_op = params.get_color_blend_operation();
    current_stroke.parameters = params.get_parameters();
    current_stroke.material = params.get_material();

    in_frame_stroke = current_stroke;
    spdlog::info("change stroke1");
}

void StrokeManager::new_history_add(std::vector<Stroke>* new_history, sToComputeStrokeData& result) {
    history = new_history;

    current_stroke.stroke_id = history->back().stroke_id + 1;
    in_frame_stroke.stroke_id = history->back().stroke_id + 1;

    redo_history.clear();

    pop_count_from_history = 0;
    redo_pop_count_from_history = 0;
    edit_list_count = 0u;

    AABB base_aabb = {};
    for (uint32_t i = 0u; i < history->size(); i++) {
        Stroke& curr_stroke = history->at(i);
        base_aabb = merge_aabbs(base_aabb, curr_stroke.get_world_AABB());
        add_stroke_to_upload_list(result.in_frame_influence, curr_stroke);
    }

    result.in_frame_stroke_aabb = base_aabb;

    result.in_frame_influence.eval_aabb_min = result.in_frame_stroke_aabb.center - result.in_frame_stroke_aabb.half_size;
    result.in_frame_influence.eval_aabb_max = result.in_frame_stroke_aabb.center + result.in_frame_stroke_aabb.half_size;
}
