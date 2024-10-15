#include "stroke_manager.h"

#include "framework/nodes/sculpt_instance.h"

#include "framework/resources/sculpt.h"

#include "spdlog/spdlog.h"
#include <glm/detail/compute_vector_relational.hpp>

void StrokeManager::init() {
    result_to_compute.set_defaults();

    edit_list.resize(EDIT_BUFFER_INITIAL_SIZE);
    result_to_compute.strokes.resize(STROKE_CONTEXT_INITIAL_SIZE);
}

void StrokeManager::add_stroke_to_upload_list(sStrokeInfluence& influence, const Stroke& stroke) {
    if (influence.strokes.size() == influence.stroke_count) {
        result_to_compute.strokes.resize(influence.strokes.size() + STROKE_CONTEXT_INCREASE);
    }

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


sStrokeInfluence* StrokeManager::undo()
{
    if (history->empty()) {
        return nullptr;
    }

    uint32_t last_stroke_id = 0u;
    uint32_t united_stroke_idx = 0;
    uint32_t last_history_index = 0u;
    float max_smooth_margin = 0.0f;

    result_to_compute.set_defaults();

    AABB in_frame_stroke_aabb;
    // Get the last stroke to undo, and compute the AABB
    for (united_stroke_idx = history->size(); united_stroke_idx > 0; --united_stroke_idx) {
        Stroke& prev = (*history)[united_stroke_idx - 1];

        // if stroke changes
        if (last_stroke_id != 0u && prev.stroke_id != last_stroke_id) {
            break;
        }

        in_frame_stroke_aabb = merge_aabbs(in_frame_stroke_aabb, prev.get_world_AABB());
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

    result_to_compute.eval_aabb_min = in_frame_stroke_aabb.center - in_frame_stroke_aabb.half_size;
    result_to_compute.eval_aabb_max = in_frame_stroke_aabb.center + in_frame_stroke_aabb.half_size;

    // Fit the AABB to the eval grid
    in_frame_stroke_aabb.half_size += glm::vec3(max_smooth_margin);
    AABB culling_aabb = compute_grid_aligned_AABB(in_frame_stroke_aabb, brick_world_size);
    in_frame_stroke_aabb.half_size -= glm::vec3(max_smooth_margin);

    // Compute and fill intersection
    compute_history_intersection(result_to_compute, culling_aabb, last_history_index);

    // In the case of first stroke, submit it as substraction to clear everything
    if (united_stroke_idx == 0) {
        Stroke prev = history->at(0);
        prev.operation = OP_SMOOTH_SUBSTRACTION;

        add_stroke_to_upload_list(result_to_compute, prev);
    } else {
        //result.in_frame_stroke = history->data()[united_stroke_idx - 1];
        add_stroke_to_upload_list(result_to_compute, history->data()[united_stroke_idx - 1]);
    }

    // Set the evaluation falg to UNDO
    result_to_compute.is_undo = 0x01u;

    return &result_to_compute;
}


sStrokeInfluence* StrokeManager::redo() {
    if (redo_history.size() <= 0u) {
        return nullptr;
    }
    uint32_t strokes_to_redo_count = redo_history.size();
    uint32_t united_stroke_idx = redo_history.back().stroke_id;
    float max_smooth_margin = 0.0f;

    result_to_compute.set_defaults();

    AABB in_frame_stroke_aabb;

    if (redo_history.size() == 1u) {
        max_smooth_margin = redo_history[0u].parameters.w;
        redo_pop_count_from_history = 1u;
        strokes_to_redo_count = 1u;
        in_frame_stroke_aabb = redo_history[0u].get_world_AABB();

        add_stroke_to_upload_list(result_to_compute, redo_history[0u]);
    } else {
        // Get the last edit to redo, and compute the AABB
        for (; strokes_to_redo_count > 0u;) {
            Stroke& curr_stroke = redo_history[strokes_to_redo_count - 1u];

            if (united_stroke_idx != curr_stroke.stroke_id) {
                break;
            }

            max_smooth_margin = glm::max(max_smooth_margin, curr_stroke.parameters.w);
            redo_pop_count_from_history++;
            strokes_to_redo_count--;
            in_frame_stroke_aabb = merge_aabbs(in_frame_stroke_aabb, curr_stroke.get_world_AABB());

            add_stroke_to_upload_list(result_to_compute, redo_history[strokes_to_redo_count]);
        }
    }
    
    spdlog::info("redo size: {}, to pop {}", redo_history.size(), redo_pop_count_from_history);


    result_to_compute.eval_aabb_min = in_frame_stroke_aabb.center - in_frame_stroke_aabb.half_size;
    result_to_compute.eval_aabb_max = in_frame_stroke_aabb.center + in_frame_stroke_aabb.half_size;

    // Fit the AABB to the eval grid
    in_frame_stroke_aabb.half_size += glm::vec3(max_smooth_margin);
    AABB culling_aabb = compute_grid_aligned_AABB(in_frame_stroke_aabb, brick_world_size);
    in_frame_stroke_aabb.half_size -= glm::vec3(max_smooth_margin);

    // Compute and fill intersection
    compute_history_intersection(result_to_compute, culling_aabb, history->size());

    return &result_to_compute;
}


sStrokeInfluence* StrokeManager::add(std::vector<Edit> new_edits) {

    result_to_compute.set_defaults();

    // Add new edits to the current stroke and the in_frame_stroke
    for (uint16_t i = 0u; i < new_edits.size(); i++) {
        in_frame_stroke.edits[in_frame_stroke.edit_count++] = new_edits[i];
    }

    // Compute AABB for the incomming strokes
    AABB in_frame_stroke_aabb = in_frame_stroke.get_world_AABB();
    AABB culling_aabb = compute_grid_aligned_AABB(in_frame_stroke_aabb, brick_world_size);
    in_frame_stroke_aabb.half_size -= glm::vec3(in_frame_stroke.parameters.w);

    // Compute and fill intersection
    compute_history_intersection(result_to_compute, culling_aabb, history->size());

    add_stroke_to_upload_list(result_to_compute, in_frame_stroke);

    for (uint16_t i = 0u; i < new_edits.size(); i++) {
        // if exceeds the maximun number of edits per stroke, store the current to the history
        // and add them to a new one, with the same ID
        if (current_stroke.edit_count == MAX_EDITS_PER_EVALUATION) {
            history->push_back(current_stroke);
            current_stroke.edit_count = 0u;
        }

        current_stroke.edits[current_stroke.edit_count++] = new_edits[i];
    }

    //if (current_stroke.edit_count > 0u) {
    //    history->push_back(current_stroke);
    //}

    redo_history.clear();

    result_to_compute.eval_aabb_min = in_frame_stroke_aabb.center - in_frame_stroke_aabb.half_size;
    result_to_compute.eval_aabb_max = in_frame_stroke_aabb.center + in_frame_stroke_aabb.half_size;

    return &result_to_compute;
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
    history = &(sculpt_instance->get_sculpt_data()->get_stroke_history());

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

sStrokeInfluence* StrokeManager::new_history_add(std::vector<Stroke>* new_history)
{
    result_to_compute.set_defaults();
    edit_list_count = 0u;

    history = new_history;

    if (history->empty()) {
        return &result_to_compute;
    }

    current_stroke.stroke_id = history->back().stroke_id + 1;
    in_frame_stroke.stroke_id = history->back().stroke_id + 1;

    redo_history.clear();

    pop_count_from_history = 0;
    redo_pop_count_from_history = 0;

    AABB base_aabb = {};
    for (uint32_t i = 0u; i < history->size(); i++) {
        Stroke& curr_stroke = history->at(i);
        base_aabb = merge_aabbs(base_aabb, curr_stroke.get_world_AABB());
        add_stroke_to_upload_list(result_to_compute, curr_stroke);
    }

    AABB in_frame_stroke_aabb = base_aabb;

    result_to_compute.eval_aabb_min = in_frame_stroke_aabb.center - in_frame_stroke_aabb.half_size;
    result_to_compute.eval_aabb_max = in_frame_stroke_aabb.center + in_frame_stroke_aabb.half_size;

    return &result_to_compute;
}


void aabb_split(AABB *container, uint32_t curr_idx, uint32_t *box_pool_top, const float half_max_division_size) {
    AABB& curr = container[curr_idx];
    const bool test_axis_x = half_max_division_size < curr.half_size.x;
    const bool test_axis_y = half_max_division_size < curr.half_size.y;
    const bool test_axis_z = half_max_division_size < curr.half_size.z;

    if (test_axis_x || test_axis_y || test_axis_z) {
        glm::vec3 half_size = curr.half_size, delta = { 0.0f, 0.0f, 0.0f };
        if (test_axis_x) {
            half_size.x /= 2.0f;
            delta = glm::vec3{ half_size.x, 0.0, 0.0 };
        } else if (test_axis_y) {
            half_size.y /= 2.0f;
            delta = glm::vec3{ 0.0, half_size.y, 0.0 };
        } else if (test_axis_z) {
            half_size.z /= 2.0f;
            delta = glm::vec3{ 0.0, 0.0, half_size.z };
        }
        spdlog::info("{} {}", * box_pool_top, half_size.x);
        uint32_t new_idx = (*box_pool_top)++;
        container[curr_idx] = { curr.center + delta, half_size };
        container[new_idx] = { curr.center - delta*2.0f, half_size};

        aabb_split(container, curr_idx, box_pool_top, half_max_division_size);
        aabb_split(container, new_idx, box_pool_top, half_max_division_size);
    }

    // recursion end
}

uint32_t StrokeManager::divide_AABB_on_max_eval_size(const AABB& base, AABB divided_bases[8]) {
    const uint32_t index_LUT[8u] = { 0u, 0u,   0u, 1u,   0u, 1u, 2u, 3u };

    uint32_t division_count = 1u;
    divided_bases[0] = base;

    // TODO send the base size via config
    const float half_max_division_size = (1.0f * AREA_MAX_EVALUATION_SIZE);

    const glm::vec3 base_min_point = base.center - base.half_size;

    aabb_split(divided_bases, 0u, &division_count, half_max_division_size);

    return division_count;
}
