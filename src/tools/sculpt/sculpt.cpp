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

bool SculptTool::update(float delta_time, StrokeParameters& stroke_parameters)
{
	Tool::update(delta_time, stroke_parameters);

	if (!enabled) return false;

	// Tool Operation changer
	if (Input::was_button_pressed(XR_BUTTON_Y))
	{
        sdOperation op = stroke_parameters.get_operation();

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

        stroke_parameters.set_operation(op);
	}

	// Sculpting (adding edits)
	if (is_tool_activated()) {

		// For debugging sculpture without a headset
		if (!Renderer::instance->get_openxr_available()) {

            //edit_to_add.position = glm::vec3(0.0);
            edit_to_add.position = glm::vec3(glm::vec3( 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1)));
			glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
			edit_to_add.dimensions = glm::vec4(0.01f, 0.01f, 0.01f, 0.01f) * 10.0f;
            //edit_to_add.dimensions = (edit_to_add.operation == OP_SUBSTRACTION) ? 3.0f * glm::vec4(0.2f, 0.2f, 0.2f, 0.2f) : glm::vec4(0.2f, 0.2f, 0.2f, 0.2f);
            edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
            // Stroke
            stroke_parameters.set_color(glm::vec4(0.1f, 0.1f, 0.1f, 1.f));
            stroke_parameters.set_primitive((random_f() > 0.25f) ? ((random_f() > 0.5f) ? SD_SPHERE : SD_CYLINDER) : SD_BOX);
            //stroke_parameters.primitive = (random_f() > 0.5f) ? SD_SPHERE : SD_BOX;
            // stroke_parameters.material = glm::vec4(random_f(), random_f(), 0.f, 0.f);
            //stroke_parameters.set_operation( (random_f() > 0.5f) ? OP_UNION : OP_SUBSTRACTION);
            stroke_parameters.set_operation(OP_SMOOTH_UNION);
            stroke_parameters.set_material_metallic(0.9);
            stroke_parameters.set_material_roughness(0.2);
            stroke_parameters.set_smooth_factor(0.01);
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
