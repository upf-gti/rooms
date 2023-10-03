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
}

void SweepTool::clean()
{

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
            fill_edits_with_stroke();

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
