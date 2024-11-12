#include "base_editor.h"

#include "engine/rooms_engine.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/container_2d.h"
#include "framework/input.h"
#include "framework/ui/io.h"

#include "graphics/renderers/rooms_renderer.h"

void BaseEditor::clean()
{

}

void BaseEditor::update(float delta_time)
{
    if (renderer->get_openxr_available()) {

        is_shift_left_pressed = Input::is_grab_pressed(HAND_LEFT);
        is_shift_right_pressed = Input::is_grab_pressed(HAND_RIGHT);

        // Update main UI

        glm::mat4x4 pose = Input::get_controller_pose(HAND_LEFT, POSE_AIM);
        pose = glm::translate(pose, glm::vec3(0.0f, 0.05f, -0.04f));
        pose = glm::rotate(pose, glm::radians(120.f), glm::vec3(1.0f, 0.0f, 0.0f));
        main_panel->set_xr_transform(Transform::mat4_to_transform(pose));

        // Update controller labels if needed

        glm::mat4x4 m = glm::rotate(glm::mat4x4(1.0f), glm::radians(60.f), glm::vec3(1.0f, 0.0f, 0.0f));
        m = glm::translate(m, glm::vec3(0.04f, -0.05f, -0.01f));

        {
            glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT);
            right_hand_box->set_xr_transform(Transform::mat4_to_transform(pose * m));
            right_hand_box->update(delta_time);
        }

        {
            pose = Input::get_controller_pose(HAND_LEFT);
            left_hand_box->set_xr_transform(Transform::mat4_to_transform(pose * m));
            left_hand_box->update(delta_time);
        }

        // Update controller speed & acceleration
        for (uint8_t i = 0; i < HAND_COUNT; ++i) {
            sControllerMovementData& data = controller_movement_data[i];
            const glm::vec3 curr_controller_pos = Input::get_controller_position(i);
            data.frame_distance = curr_controller_pos - data.prev_position;
            const glm::vec3 curr_controller_velocity = (data.frame_distance) / delta_time;
            data.acceleration = (curr_controller_velocity - data.velocity) / delta_time;
            data.velocity = curr_controller_velocity;
            data.prev_position = curr_controller_pos;
        }

        // Manage auto-hide of the menu in XR
        if (!is_something_hovered()) {
            last_hover_time += delta_time;

            if (last_hover_time > 10.0f) {
                main_panel->set_visibility(false);
            }
        }
        else {
            last_hover_time = 0.0f;
        }

        float min_acceleration_trigger = 40.0f;

        if (!main_panel->get_visibility() && glm::length(controller_movement_data[HAND_LEFT].acceleration) > min_acceleration_trigger) {
            main_panel->set_visibility(true);
            last_hover_time = 0.0f;
            Engine::instance->vibrate_hand(HAND_LEFT, 0.25f, 0.25f);
        }
    }

    main_panel->update(delta_time);
}

void BaseEditor::render()
{
    // Shortcuts panels
    if (renderer->get_openxr_available()) {
        left_hand_box->render();
        right_hand_box->render();
    }

    main_panel->render();
}

void BaseEditor::update_shortcuts(const std::unordered_map<uint8_t, bool>& active_shortcuts)
{
    assert(renderer->get_openxr_available() && "Updating shortcuts in non-XR mode!");

    glm::mat4x4 m = glm::rotate(glm::mat4x4(1.0f), glm::radians(-120.f), glm::vec3(1.0f, 0.0f, 0.0f));
    m = glm::translate(m, glm::vec3(0.02f, 0.0f, 0.02f));

    glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT);
    right_hand_box->get_xr_viewport()->set_transform(Transform::mat4_to_transform(pose * m));

    pose = Input::get_controller_pose(HAND_LEFT);
    left_hand_box->get_xr_viewport()->set_transform(Transform::mat4_to_transform(pose * m));

    /*
        Change label visibility
    */

    // Left controller
    for (auto label_ptr : left_hand_box->get_children()) {
        ui::ImageLabel2D* label = dynamic_cast<ui::ImageLabel2D*>(label_ptr);
        assert(label);
        uint8_t mask = label->get_mask();
        bool value = active_shortcuts.contains(mask) ? active_shortcuts.at(mask) : false;
        label->set_visibility(value);
    }

    // Right controller
    for (auto label_ptr : right_hand_box->get_children()) {
        ui::ImageLabel2D* label = dynamic_cast<ui::ImageLabel2D*>(label_ptr);
        assert(label);
        uint8_t mask = label->get_mask();
        bool value = active_shortcuts.contains(mask) ? active_shortcuts.at(mask) : false;
        label->set_visibility(value);
    }
}

bool BaseEditor::is_something_hovered()
{
    if (!IO::any_hover()) {
        return false;
    }

    auto xr_panel = dynamic_cast<ui::XRPanel*>(IO::get_hover());

    return !xr_panel || (xr_panel && xr_panel->get_is_button());
}

bool BaseEditor::is_something_focused()
{
    if (!IO::any_focus()) {
        return false;
    }

    auto xr_panel = dynamic_cast<ui::XRPanel*>(IO::get_focus());

    return !xr_panel || (xr_panel && xr_panel->get_is_button());
}
