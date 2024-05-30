#include "base_editor.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"

#include "engine/rooms_engine.h"

void BaseEditor::clean()
{
    _DESTROY_(right_hand_ui_3D);
    _DESTROY_(left_hand_ui_3D);
}

void BaseEditor::update(float delta_time)
{
    if (main_panel_3d) {
        glm::mat4x4 pose = Input::get_controller_pose(HAND_LEFT, POSE_AIM);
        pose = glm::rotate(pose, glm::radians(-45.f), glm::vec3(1.0f, 0.0f, 0.0f));
        main_panel_3d->set_model(pose);
    }
    else {
        main_panel_2d->update(delta_time);
    }
}

void BaseEditor::render()
{
    // Render 3d or 2d panels
    if (main_panel_3d) {
        main_panel_3d->render();
    }
    else {
        main_panel_2d->render();
    }
}
