#include "tutorial_editor.h"

#include "framework/nodes/panel_2d.h"
#include "framework/input.h"
#include "framework/math/math_utils.h"

#include "graphics/renderers/rooms_renderer.h"

#include "engine/rooms_engine.h"

#include "glm/gtx/quaternion.hpp"

void TutorialEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    // Create tutorial/welcome panel
    {
        panel = new Node2D("tutorial_root", { 0.0f, 0.0f }, { 1.0f, 1.0f });

        panels[TUTORIAL_WELCOME] = generate_panel("root_welcome", "data/textures/tutorial/welcome_screen.png", TUTORIAL_NONE, TUTORIAL_SCENE_1);
        panels[TUTORIAL_WELCOME]->set_visibility(true);

        panels[TUTORIAL_SCENE_1] = generate_panel("root_scene_1", "data/textures/tutorial/scene_1.png", TUTORIAL_WELCOME, TUTORIAL_SCENE_2);
        panels[TUTORIAL_SCENE_2] = generate_panel("root_scene_2", "data/textures/tutorial/scene_2.png", TUTORIAL_SCENE_1, TUTORIAL_SCENE_3);
        panels[TUTORIAL_SCENE_3] = generate_panel("root_scene_3", "data/textures/tutorial/scene_3.png", TUTORIAL_SCENE_2, TUTORIAL_STAMP_SMEAR);
        panels[TUTORIAL_STAMP_SMEAR] = generate_panel("root_stamp_smear", "data/textures/tutorial/stamp_smear.png", TUTORIAL_SCENE_3, TUTORIAL_PRIMITIVES_OPERATIONS);
        panels[TUTORIAL_PRIMITIVES_OPERATIONS] = generate_panel("root_primitives_op", "data/textures/tutorial/prims_ops.png", TUTORIAL_STAMP_SMEAR, TUTORIAL_CURVES);
        panels[TUTORIAL_CURVES] = generate_panel("root_curves", "data/textures/tutorial/curves.png", TUTORIAL_PRIMITIVES_OPERATIONS, TUTORIAL_GUIDES);
        panels[TUTORIAL_GUIDES] = generate_panel("root_guides", "data/textures/tutorial/guides.png", TUTORIAL_CURVES, TUTORIAL_MATERIAL);
        panels[TUTORIAL_MATERIAL] = generate_panel("root_materials", "data/textures/tutorial/materials.png", TUTORIAL_GUIDES, TUTORIAL_PAINT);
        panels[TUTORIAL_PAINT] = generate_panel("root_paint", "data/textures/tutorial/paint.png", TUTORIAL_MATERIAL, TUTORIAL_UNDO_REDO);
        panels[TUTORIAL_UNDO_REDO] = generate_panel("root_undo_redo", "data/textures/tutorial/undo_redo.png", TUTORIAL_PAINT, TUTORIAL_NONE);
    }
}

void TutorialEditor::clean()
{
    BaseEditor::clean();
}

void TutorialEditor::update(float delta_time)
{
    if(renderer->get_openxr_available()) {
        glm::mat4x4 m(1.0f);
        glm::vec3 eye = renderer->get_camera_eye();
        glm::vec3 new_pos = eye + renderer->get_camera_front() * 1.25f;

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
        m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

        panel->set_xr_transform(Transform::mat4_to_transform(m));
    }

    panel->update(delta_time);
}

void TutorialEditor::render()
{
    panel->render();
}

ui::XRPanel* TutorialEditor::generate_panel(const std::string& name, const std::string& path, uint8_t prev, uint8_t next)
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height)) * 0.5f;
    glm::vec2 pos = size * 0.5f;

    if (renderer->get_openxr_available()) {
        size = glm::vec2(1920.f, 1080.0f);
        pos = -size * 0.5f;
    }

    const glm::vec2& button_size = { size.x * 0.4f, size.y * 0.25f };
    const glm::vec2& mini_button_size = { size.x * 0.2f, size.y * 0.125f };

    ui::XRPanel* new_panel = new ui::XRPanel(name, path, pos, size);
    new_panel->set_visibility(false);
    panel->add_child(new_panel);

    if (prev != TUTORIAL_NONE) {
        new_panel->add_button(name + "_prev", "data/textures/tutorial/back.png", { size.x * 0.25f, size.y - mini_button_size.y * 1.2f }, mini_button_size);
        Node::bind(name + "_prev", [&, c = new_panel, p = prev](const std::string& signal, void* button) {
            c->set_visibility(false);
            panels[p]->set_visibility(true);
        });
    }

    if (next != TUTORIAL_NONE) {
        new_panel->add_button(name + "_next", "data/textures/tutorial/next.png", { size.x * 0.75f, size.y - mini_button_size.y * 1.2f }, mini_button_size);
        Node::bind(name + "_next", [&, c = new_panel, n = next](const std::string& signal, void* button) {
            c->set_visibility(false);
            panels[n]->set_visibility(true);
            // If in scene editor, close the tutorial, that will reopen when entering the sculpt editor
            auto engine = static_cast<RoomsEngine*>(Engine::instance);
            if (engine->get_current_editor_type() == SCENE_EDITOR && n == TUTORIAL_STAMP_SMEAR) {
                engine->toggle_tutorial();
            }
        });
    }
    // close button
    else {
        new_panel->add_button(name + "_next", "data/textures/tutorial/close.png", { size.x * 0.75f, size.y - mini_button_size.y * 1.2f }, mini_button_size);
        Node::bind(name + "_next", [&, c = new_panel, n = next](const std::string& signal, void* button) {
            auto engine = static_cast<RoomsEngine*>(Engine::instance);
            engine->toggle_tutorial();
        });
    }

    return new_panel;
}
