#include "tool.h"
#include "utils.h"
#include "framework/input.h"
#include "sculpt_editor.h"

void Tool::initialize()
{
}

void Tool::clean()
{

}

bool Tool::update(float delta_time, StrokeParameters& stroke_parameters)
{
    edit_update_counter += delta_time;

    // Update edit transform
    edit_to_add.position = Input::get_controller_position(HAND_RIGHT);
    edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT));

    return true;
}

bool Tool::use_tool()
{
    if (edit_to_add.weigth_difference(previous_edit) > 0.003f) {
        previous_edit = edit_to_add;
        return true;
    }

    return false;
}
