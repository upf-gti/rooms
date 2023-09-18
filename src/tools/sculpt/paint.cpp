#include "paint.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"

void PaintTool::initialize()
{
    Tool::initialize();

    edit_to_add.operation = OP_SMOOTH_PAINT;
}

void PaintTool::clean()
{

}

bool PaintTool::update(float delta_time)
{
	Tool::update(delta_time);

    if (!enabled) return false;

    // Tool Operation changer
    if (Input::was_button_pressed(XR_BUTTON_Y))
    {
        edit_to_add.operation = edit_to_add.operation == OP_PAINT ? OP_SMOOTH_PAINT : OP_PAINT;
    }

	if (is_tool_activated())
	{
        return use_tool();
	}

    return false;
}

void PaintTool::render_scene()
{
    if (!enabled) return;
}

void PaintTool::render_ui()
{
	if (!enabled) return;
}

bool PaintTool::use_tool()
{
	if (Tool::use_tool()) {
		return true;
	}

	return false;
}
