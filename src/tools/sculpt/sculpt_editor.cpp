#include "sculpt_editor.h"

#include "sculpt.h"
#include "paint.h"

#include "graphics/raymarching_renderer.h"

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RaymarchingRenderer*>(Renderer::instance);

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    tool_controller.set_workspace({ 88.f, 48.f }, XR_BUTTON_A, POSE_AIM);

    // Tools Menu
    {
        tool_controller.make_button("sculpt_mode", { 8.f, 8.f }, { 32.f, 32.f }, colors::WHITE, "data/textures/sculpt.png");
        tool_controller.make_button("color_mode", { 48.f , 8.f }, { 32.f, 32.f }, colors::WHITE, "data/textures/paint.png");
    }

    // Events
    {
        tool_controller.connect("sculpt_mode", [&](const std::string& signal, float value) {
            enable_tool(SCULPT);
        });
        tool_controller.connect("color_mode", [&](const std::string& signal, float value) {
            enable_tool(PAINT);
        });
    }

    enable_tool(SCULPT);
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    bool tool_used = tools[current_tool]->update(delta_time);

    if (current_tool == SCULPT && tool_used) {
        sculpt_started = true;
    }

    Edit& edit_to_add = tools[current_tool]->get_edit_to_add();

    // Set center of sculpture
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);
    }

    // Rotate the scene TODO: when ready move this out of tool to engine
    if (is_rotation_being_used()) {

        if (!rotation_started) {
            initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
            initial_hand_translation = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f);
        }

        rotation_diff = glm::inverse(initial_hand_rotation) * glm::inverse(Input::get_controller_rotation(HAND_LEFT));
        translation_diff = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f) - initial_hand_translation;

        renderer->set_sculpt_rotation(sculpt_rotation * rotation_diff);
        renderer->set_sculpt_start_position(sculpt_start_position + translation_diff);

        rotation_started = true;
    }
    else {
        if (rotation_started) {
            sculpt_rotation = sculpt_rotation * rotation_diff;
            sculpt_start_position = sculpt_start_position + translation_diff;
            rotation_started = false;
            rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
        }
    }

    // Set position of the preview edit
    renderer->set_preview_edit(edit_to_add);

    if (tool_used) {
        renderer->push_edit(edit_to_add);

        // If the mirror is activated, mirror using the plane, and add another edit to the list
        if (use_mirror) {
            float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
            edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

            renderer->push_edit(edit_to_add);
        }
    }

    tool_controller.update(delta_time);

    //if (Input::is_button_pressed(XR_BUTTON_B))
    //{
    //    tool_controller.enabled = true;

    //    for (auto& tool : tools)
    //    {
    //        if (tool)
    //        {
    //            tool->stop();
    //        }
    //    }
    //}
}

void SculptEditor::render()
{
#ifdef XR_SUPPORT
    if (current_tool != NONE) {
        tools[current_tool]->render_scene();
        tools[current_tool]->render_ui();
    }
    tool_controller.render();
#endif
}

void SculptEditor::enable_tool(eTool tool)
{
    tool_controller.enabled = false;

    if (current_tool != NONE) {
        tools[current_tool]->stop();
    }

    tools[tool]->start();
    current_tool = tool;
}
