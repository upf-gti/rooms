#include "sweep.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/intersections.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"
#include "sculpt_editor.h"

void SweepTool::initialize()
{
    Tool::initialize();

    arc_length_LUT = new float[STARTING_ARC_LENGTH];
    arc_length_LUT_storeage_size = STARTING_ARC_LENGTH;
}

void SweepTool::clean()
{
    delete arc_length_LUT;
}

// Curve functions =============
inline glm::vec3 SweepTool::sBezierCurve::evaluate(const float t) const {
    const float t_inv = (1.0f - t);
    return t_inv * t_inv * start + 2.0f * t * t_inv * control + t * t * end;
}

inline void SweepTool::fill_arc_length_LUT(const uint32_t element_count, const float bowstring_length) {
    // Segmentate the curve and fill teh LUT with the aproximated lengt of the segments
    const float step = 1.0f / (element_count);
    float current_length = 0.0f;
    glm::vec3 prev_pos = curve.evaluate(0.0f);
    arc_length_LUT_size = 0u;

    for (float i = 1.0f; i < element_count; i++) {
        const glm::vec3 curr_pos = curve.evaluate(i * step);

        const glm::vec3 delta_pos = prev_pos - curr_pos;

        current_length += glm::length(delta_pos);

        arc_length_LUT[arc_length_LUT_size++] = current_length;

        prev_pos = curr_pos;
    }

    curve_length = current_length;
}

// TODO: This can be much faster, with a binary search
uint32_t SweepTool::get_closest_arc_length(const float length) const {
    uint32_t i = 1u;

    for (; i < arc_length_LUT_size; i++) {
        if (arc_length_LUT[i] > length) {
            return i-1u;
        }
    }
    return 0u;
}

// Fin the closest segment based on length and interpolate
inline float SweepTool::aprox_inverse_curve_length(const float length) const {
    const float target_length = length * arc_length_LUT[arc_length_LUT_size-1u];

    const uint32_t closest_index = get_closest_arc_length(target_length);

    const float clossest_length = arc_length_LUT[closest_index];

    if (clossest_length < target_length) {
        return (closest_index + (target_length - clossest_length) / (arc_length_LUT[closest_index + 1u] - clossest_length)) / arc_length_LUT_size;
    } else {
        return closest_index / length;
    }
}

void SweepTool::fill_edits_with_arc(const float delta) {
    const glm::vec3 stroke_end_position = Input::get_controller_position(HAND_RIGHT);
    const glm::vec3 stroke_direction = glm::normalize(stroke_end_position - stroke_start_position);

    const float stroke_length = glm::length(stroke_end_position - stroke_start_position);
    const glm::vec3 half_point = stroke_start_position + stroke_direction * (stroke_length * 0.5f);

    // If the current stroke segment is too small, just do a edit line
    if (stroke_length < 0.02f) {
        fill_edits_with_stroke();
        return;
    }

    // Compute control point, using a plane intersection in the middle of the segment, and a ray from the controller
    const glm::vec3 ray_direction = Input::get_controller_rotation(HAND_RIGHT, POSE_AIM) * glm::vec3(0.0f, 0.0f, -1.0f);
    float dist = 0.0f;
    intersection::ray_plane(stroke_end_position,
                            ray_direction,
                            half_point,
                            stroke_direction,
                            dist);

    glm::vec3 control_point = ray_direction * dist + stroke_end_position;

    // Clamp the control point to a certain radius (half a meter)
    glm::vec3 plane_center_to_control_point = control_point - half_point;
    float center_to_control_distance = glm::length(plane_center_to_control_point);
    center_to_control_distance = glm::sign(center_to_control_distance) * glm::min(glm::abs(center_to_control_distance), 0.50f);
    control_point = glm::normalize(plane_center_to_control_point) * center_to_control_distance + half_point;

    // Set control point of the curve to the plane intersection
    curve.set_curve(stroke_start_position, control_point, stroke_end_position);

    // Calculate the number of edits
    // Calculate the length of the curve with a 5 segment aproximation
    fill_arc_length_LUT(5u, stroke_length);
    const float aprox_curve_length = curve_length;

    uint32_t number_of_edits = (uint32_t)glm::ceil(aprox_curve_length / inter_edit_distance);
    number_of_edits += (number_of_edits % 2u != 0u) ? 1u : 0u; // Always a even number

    // If we need more edits, reescale the buffer
    if (number_of_edits >= arc_length_LUT_storeage_size) {
        delete arc_length_LUT;
        arc_length_LUT_storeage_size += ARC_LENGTH_INCREASE;
        arc_length_LUT = new float[arc_length_LUT_storeage_size];
    }

    // Compute the table with the aprozximated number of edits
    fill_arc_length_LUT(number_of_edits, stroke_length);

    // Add the edits
    tmp_edit_storage.clear();
    glm::vec4 end_edit_dimensions = edit_to_add.dimensions;
    for (float i = 2.0f; i < number_of_edits-1; i++) {
        const float t = aprox_inverse_curve_length(i / number_of_edits);

        edit_to_add.position = curve.evaluate(t);
        edit_to_add.dimensions = glm::mix(start_edit_dimensions, end_edit_dimensions, t);
        tmp_edit_storage.push_back(edit_to_add);
    }
}


uint32_t SweepTool::get_number_of_edits_in_stroke(const glm::vec3 &stroke_end_position, const glm::quat &stroke_end_orientation) const {
    return (uint32_t) glm::ceil(glm::length(stroke_start_position - stroke_end_position) / inter_edit_distance);
}

void SweepTool::fill_edits_with_stroke() {
    const glm::vec3 stroke_end_position = Input::get_controller_position(HAND_RIGHT);

    const glm::vec3 stroke_direction = glm::normalize(stroke_end_position - stroke_start_position);

    edit_to_add.position = stroke_start_position;

    // mmm not really happy with this
    tmp_edit_storage.clear();
    tmp_edit_storage.push_back(edit_to_add);

    for (uint32_t edit_count = get_number_of_edits_in_stroke(stroke_end_position, {0.0f,0.0f,0.0f,1.0f }); edit_count > 0u; edit_count--) {
        edit_to_add.position += inter_edit_distance * stroke_direction;
        tmp_edit_storage.push_back(edit_to_add);
    }
}

bool SweepTool::update(float delta_time)
{
	Tool::update(delta_time);

	if (!enabled) return false;

	// Tool Operation changer
	if (Input::was_button_pressed(XR_BUTTON_Y))
	{
        sdOperation& op = stroke_parameters.operation;

		switch (op)
		{
		case OP_UNION:
            op = OP_SUBSTRACTION;
			break;
		case OP_SUBSTRACTION:
            op = OP_UNION;
            break;
		case OP_SMOOTH_UNION:
            op = OP_SMOOTH_SUBSTRACTION;
			break;
		case OP_SMOOTH_SUBSTRACTION:
            op = OP_SMOOTH_UNION;
            break;
		default:
			break;
		}
	}

    // Set the inter-edit distance TODO: How and where? For now in here
    const float size_multiplier = Input::get_thumbstick_value(HAND_RIGHT).x * delta_time * 0.1f;
    inter_edit_distance = glm::clamp(size_multiplier + inter_edit_distance, 0.006f, 0.1f);

    glm::vec3 controller_position = Input::get_controller_position(HAND_RIGHT);

	// Sculpting (adding edits)
    stroke_prev_state = stroke_state;
	if (use_tool()) {
        if (stroke_prev_state == NO_STROKE && stroke_state == IN_STROKE) {
            stroke_start_position = controller_position;
            start_edit_dimensions = edit_to_add.dimensions;
            return false;
        }

        if (stroke_prev_state == IN_STROKE && stroke_state == IN_STROKE) {
            fill_edits_with_arc(delta_time);

            sculpt_editor->add_preview_edit_list(tmp_edit_storage);
            return false;
        }
    } else {
        if (stroke_prev_state == IN_STROKE && stroke_state == NO_STROKE) {
            fill_edits_with_arc(delta_time);
            sculpt_editor->add_edit_list(tmp_edit_storage);
            return true;
        }
    }

    return false;
}

void SweepTool::render_scene()
{
    if (!enabled) return;
}

void SweepTool::render_ui()
{
	if (!enabled) return;
}

bool SweepTool::use_tool()
{
    if (is_tool_activated()) {
        stroke_state = IN_STROKE;
		return true;
	}

    stroke_state = NO_STROKE;
	return false;
}
