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

bool Tool::update(float delta_time)
{
    edit_update_counter += delta_time;

    // Update edit position
    glm::vec3 controller_pos_perfect = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
    edit_to_add.position += (controller_pos_perfect - edit_to_add.position) * 0.50f;
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
