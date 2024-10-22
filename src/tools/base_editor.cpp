#include "base_editor.h"

#include "engine/rooms_engine.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/container_2d.h"
#include "framework/input.h"

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
            right_hand_box->get_xr_viewport()->set_transform(Transform::mat4_to_transform(pose * m));
            right_hand_box->get_xr_viewport()->update(delta_time);
        }

        {
            pose = Input::get_controller_pose(HAND_LEFT);
            left_hand_box->get_xr_viewport()->set_transform(Transform::mat4_to_transform(pose * m));
            left_hand_box->get_xr_viewport()->update(delta_time);
        }
    }

    main_panel->update(delta_time);
}

void BaseEditor::render()
{
    // Shortcuts panels
    if (renderer->get_openxr_available()) {
        left_hand_box->get_xr_viewport()->render();
        left_hand_box->get_xr_viewport()->render();
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
