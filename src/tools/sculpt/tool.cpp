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
    edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
    edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT));

    // Update common edit dimensions
    float size_multipler = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 0.1f;
    glm::vec3 new_dimensions = glm::clamp(size_multipler + glm::vec3(edit_to_add.dimensions), 0.001f, 0.1f);
    edit_to_add.dimensions = glm::vec4(new_dimensions, edit_to_add.dimensions.w);

    // Update primitive specific size
    size_multipler = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1f;
    edit_to_add.dimensions.w = glm::clamp(size_multipler + edit_to_add.dimensions.w, 0.001f, 0.1f);

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
