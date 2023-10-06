#include "sweep.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"
#include "sculpt_editor.h"

void SweepTool::initialize()
{
    Tool::initialize();

    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    arc_length_LUT = new float[STARTING_ARC_LENGTH];
    arc_length_LUT_storeage_size = STARTING_ARC_LENGTH;
}

void SweepTool::clean()
{

}


// Curve functions =============
inline glm::vec3 SweepTool::sBezierCurve::evaluate(const float t) const {
    const float t_inv = (1.0f - t);
    return t_inv * t_inv * start + 2.0f * t * t_inv * control + t * t * end;
}

inline void SweepTool::fill_arc_length_LUT(const uint32_t element_count, const float bowstring_length) {
    const float step = 1.0f / (element_count);
    float current_length = 0.0f;// prev_x = curve.x(0.0f), prev_y = curve.y(0.0f);
    glm::vec3 prev_pos = curve.evaluate(0.0f);
    arc_length_LUT_size = 0u;

    for (float i = 1.0f; i < element_count; i++) {
        const glm::vec3 curr_pos = curve.evaluate(i * step);

        const glm::vec3 delta_pos = prev_pos - curr_pos;

        current_length += glm::length(delta_pos);

        arc_length_LUT[arc_length_LUT_size++] = current_length;

        prev_pos = curr_pos;
    }
}

uint32_t SweepTool::get_closest_arc_length(const float length) const {
    uint32_t i = 1u;

    for (; i < arc_length_LUT_size; i++) {
        if (arc_length_LUT[i] > length) {
            return i-1u;
        }
    }
    return 0u;
}


inline float SweepTool::aprox_inverse_curve_length(const float length) const {
    const float target_length = length * arc_length_LUT[arc_length_LUT_size-1u];

    const uint32_t closest_index = get_closest_arc_length(target_length);

    const float clossest_length = arc_length_LUT[closest_index];

    return (closest_index + (target_length - clossest_length) / (arc_length_LUT[closest_index + 1u] - clossest_length)) / arc_length_LUT_size;
}

void SweepTool::fill_edits_with_arc() {
    const glm::vec3 stroke_end_position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 stroke_direction = glm::normalize(stroke_end_position - stroke_start_position);
    const glm::vec3 segment_normal = glm::cross(stroke_direction, Input::get_controller_rotation(HAND_RIGHT) * stroke_direction);

    const float stroke_length = glm::length(stroke_end_position - stroke_start_position);
    const glm::vec3 half_point = stroke_start_position + stroke_direction * (stroke_length * 0.5f);
    const uint32_t number_of_edits = (uint32_t)glm::ceil(stroke_length / inter_edit_distance);

    curve.set_curve(stroke_start_position, segment_normal * stroke_length + half_point, stroke_end_position);
    fill_arc_length_LUT(number_of_edits, stroke_length);
    tmp_edit_storage.clear();

    for (float i = 1.0f; i < number_of_edits; i++) {
        const float t = aprox_inverse_curve_length(i / number_of_edits);

        edit_to_add.position = curve.evaluate(t);
        tmp_edit_storage.push_back(edit_to_add);
    }
}


uint32_t SweepTool::get_number_of_edits_in_stroke(const glm::vec3 &stroke_end_position, const glm::quat &stroke_end_orientation) const {
    return (uint32_t) glm::ceil(glm::length(stroke_start_position - stroke_end_position) / inter_edit_distance);
}

void SweepTool::fill_edits_with_stroke() {
    const glm::vec3 stroke_end_position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);

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
        sdOperation& op = edit_to_add.operation;

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

    glm::vec3 controller_position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);

	// Sculpting (adding edits)
    stroke_prev_state = stroke_state;
	if (use_tool()) {

        if (stroke_prev_state == NO_STROKE && stroke_state == IN_STROKE) {
            stroke_start_position = controller_position;
            return false;
        }

        if (stroke_prev_state == IN_STROKE && stroke_state == IN_STROKE) {
            fill_edits_with_arc();

            for(uint32_t i = 0u; i < tmp_edit_storage.size(); i++) {
                renderer->add_preview_edit(tmp_edit_storage[i]);
            }
            return false;
        }

        if (stroke_prev_state == IN_STROKE && stroke_state == NO_STROKE) {
            // Submit tmp_edit_list
            //return true;
            renderer->push_edit_list(tmp_edit_storage);
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
