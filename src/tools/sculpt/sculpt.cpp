#include "sculpt.h"
#include "utils.h"
#include "framework/input.h"
#include "sculpt_editor.h"

#include "graphics/renderer.h"

void SculptTool::initialize()
{
    Tool::initialize();
}

void SculptTool::clean()
{

}

bool SculptTool::update(float delta_time)
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

        stroke_parameters.was_operation_changed = true;
	}

	// Sculpting (adding edits)
	if (is_tool_activated()) {

		// For debugging sculpture without a headset
		if (!Renderer::instance->get_openxr_available()) {

            //edit_to_add.position = glm::vec3(0.0);
            edit_to_add.position = glm::vec3(glm::vec3( 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1)));
			glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
			edit_to_add.dimensions = glm::vec4(0.01f, 0.01f, 0.01f, 0.01f);
			edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
            // Stroke
            stroke_parameters.color = glm::vec4(random_f(), random_f(), random_f(), 0.f);
            stroke_parameters.operation = (random_f() > 0.5f) ? OP_UNION : OP_SUBSTRACTION;
            //edit_to_add.dimensions = (edit_to_add.operation == OP_SUBSTRACTION) ? 3.0f * glm::vec4(0.2f, 0.2f, 0.2f, 0.2f) : glm::vec4(0.2f, 0.2f, 0.2f, 0.2f);
            stroke_parameters.was_operation_changed = true;
		}

        return use_tool();
	}

    return false;
}

void SculptTool::render_scene()
{
    if (!enabled) return;
}

void SculptTool::render_ui()
{
	if (!enabled) return;
}

bool SculptTool::use_tool()
{
	if (Tool::use_tool()) {
		return true;
	}

	return false;
}
