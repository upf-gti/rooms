#include "sculpt.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"
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

	// Sculpting (adding edits)
	if (is_tool_activated()) {

		// For debugging sculpture without a headset
		if (!Renderer::instance->get_openxr_available()) {
			edit_to_add.position = glm::vec3(0.5f * (random_f() * 2 - 1), 0.5f * (random_f() * 2 - 1), 0.5f * (random_f() * 2 - 1));
			glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
			edit_to_add.dimensions = glm::vec4(0.025f, 0.025f, 0.025f, 0.025f);
			edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
            edit_to_add.operation = random_f() > 0.5 ? OP_SMOOTH_UNION : OP_SMOOTH_SUBSTRACTION;
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
