#include "base_editor.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"

#include "graphics/renderers/rooms_renderer.h"

#include "engine/rooms_engine.h"

void BaseEditor::clean()
{
    _DESTROY_(right_hand_ui_3D);
    _DESTROY_(left_hand_ui_3D);
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
        main_panel_3d->set_transform(Transform::mat4_to_transform(pose));
        main_panel_3d->update(delta_time);

        // Update controller labels if needed

        glm::mat4x4 m = glm::rotate(glm::mat4x4(1.0f), glm::radians(60.f), glm::vec3(1.0f, 0.0f, 0.0f));
        m = glm::translate(m, glm::vec3(0.04f, -0.05f, -0.01f));

        if (right_hand_ui_3D) {
            glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT);
            right_hand_ui_3D->set_transform(Transform::mat4_to_transform(pose * m));
            right_hand_ui_3D->update(delta_time);
        }

        if (left_hand_ui_3D) {
            pose = Input::get_controller_pose(HAND_LEFT);
            left_hand_ui_3D->set_transform(Transform::mat4_to_transform(pose * m));
            left_hand_ui_3D->update(delta_time);
        }
    }
    else {
        main_panel_2d->update(delta_time);
    }
}

void BaseEditor::render()
{
    // Render 3d or 2d panels

    if (renderer->get_openxr_available()) {

        main_panel_3d->render();

        if (right_hand_ui_3D) {
            right_hand_ui_3D->render();
        }

        if (left_hand_ui_3D) {
            left_hand_ui_3D->render();
        }
    }
    else {
        main_panel_2d->render();
    }
}

void BaseEditor::update_shortcuts(const std::unordered_map<uint8_t, bool>& active_shortcuts)
{
    glm::mat4x4 m = glm::rotate(glm::mat4x4(1.0f), glm::radians(-120.f), glm::vec3(1.0f, 0.0f, 0.0f));
    m = glm::translate(m, glm::vec3(0.02f, 0.0f, 0.02f));

    glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT);
    right_hand_ui_3D->set_transform(Transform::mat4_to_transform(pose * m));

    pose = Input::get_controller_pose(HAND_LEFT);
    left_hand_ui_3D->set_transform(Transform::mat4_to_transform(pose * m));

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
