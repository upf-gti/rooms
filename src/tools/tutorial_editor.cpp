#include "tutorial_editor.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"

#include "graphics/renderers/rooms_renderer.h"

#include "engine/rooms_engine.h"

#include "glm/gtx/quaternion.hpp"

void TutorialEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    // Create tutorial/welcome panel
    {
        xr_panel_2d = new Node2D("tutorial_root", { 0.0f, 0.0f }, { 1.0f, 1.0f });

        auto webgpu_context = Renderer::instance->get_webgpu_context();
        glm::vec2 size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height)) * 0.5f;
        glm::vec2 pos = size * 0.5f;

        if (renderer->get_openxr_available()) {
            size = glm::vec2(1920.f, 1080.0f);
            pos = -size * 0.5f;
        }

        const glm::vec2& button_size = { size.x * 0.4f, size.y * 0.25f };
        const glm::vec2& mini_button_size = { size.x * 0.2f, size.y * 0.125f };

        // Welcome panel
        {
            ui::XRPanel* welcome_panel = new ui::XRPanel("root_welcome", "data/textures/tutorial/welcome_screen.png", pos, size);
            xr_panel_2d->add_child(welcome_panel);
            welcome_panel->add_button("start_tutorial", "data/textures/tutorial/start_tutorial.png", { button_size.x * 0.75f, button_size.y }, button_size);
            welcome_panel->add_button("create_button", "data/textures/tutorial/create_now.png", { size.x - button_size.x * 0.75f, button_size.y }, button_size);
            panels[TUTORIAL_WELCOME] = welcome_panel;

            Node::bind("create_button", [&](const std::string& signal, void* button) { RoomsEngine::switch_editor(SCENE_EDITOR); });
            Node::bind("start_tutorial", [&](const std::string& signal, void* button) {
                panels[TUTORIAL_WELCOME]->set_visibility(false);
                panels[TUTORIAL_STAMP_SMEAR]->set_visibility(true);
            });
        }

        panels[TUTORIAL_STAMP_SMEAR] = generate_panel("root_stamp_smear", "data/textures/tutorial/stamp_smear.png", TUTORIAL_NONE, TUTORIAL_PRIMITIVES_OPERATIONS);
        panels[TUTORIAL_PRIMITIVES_OPERATIONS] = generate_panel("root_primitives_op", "data/textures/tutorial/prims_ops.png", TUTORIAL_STAMP_SMEAR, TUTORIAL_CURVES);
        panels[TUTORIAL_CURVES] = generate_panel("root_curves", "data/textures/tutorial/curves.png", TUTORIAL_PRIMITIVES_OPERATIONS, TUTORIAL_GUIDES);
        panels[TUTORIAL_GUIDES] = generate_panel("root_guides", "data/textures/tutorial/guides.png", TUTORIAL_CURVES, TUTORIAL_MATERIAL);
        panels[TUTORIAL_MATERIAL] = generate_panel("root_materials", "data/textures/tutorial/materials.png", TUTORIAL_GUIDES, TUTORIAL_PAINT);
        panels[TUTORIAL_PAINT] = generate_panel("root_paint", "data/textures/tutorial/paint.png", TUTORIAL_MATERIAL, TUTORIAL_UNDO_REDO);
        panels[TUTORIAL_UNDO_REDO] = generate_panel("root_undo_redo", "data/textures/tutorial/undo_redo.png", TUTORIAL_PAINT, TUTORIAL_NONE);

        if (renderer->get_openxr_available()) {
            xr_panel_3d = new Viewport3D(xr_panel_2d);
            xr_panel_3d->set_active(true);
        }
    }
}

void TutorialEditor::clean()
{
    BaseEditor::clean();
}

void TutorialEditor::update(float delta_time)
{
    if (xr_panel_3d) {

        // Update welcome screen following headset??

        glm::mat4x4 m(1.0f);
        glm::vec3 eye = renderer->get_camera_eye();
        glm::vec3 new_pos = eye + renderer->get_camera_front() * 1.25f;

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
        m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

        xr_panel_3d->set_transform(Transform::mat4_to_transform(m));
        xr_panel_3d->update(delta_time);
    }
    else {
        xr_panel_2d->update(delta_time);
    }
}

void TutorialEditor::render()
{
    // RoomsEngine::render_controllers();

    if (xr_panel_3d) {
        xr_panel_3d->render();
    }
    else {
        xr_panel_2d->render();
    }
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
    xr_panel_2d->add_child(new_panel);

    if (prev != TUTORIAL_NONE) {
        new_panel->add_button(name + "_prev", "data/textures/tutorial/back.png", { size.x * 0.25f, mini_button_size.y }, mini_button_size);
        Node::bind(name + "_prev", [&, c = new_panel, p = prev](const std::string& signal, void* button) {
            c->set_visibility(false);
            panels[p]->set_visibility(true);
        });
    }

    // There's always next button.. next panel or start creating!
    new_panel->add_button(name + "_next", "data/textures/tutorial/next.png", { size.x * 0.75f, mini_button_size.y }, mini_button_size);

    Node::bind(name + "_next", [&, c = new_panel, n = next](const std::string& signal, void* button) {
        c->set_visibility(false);
        if (n != TUTORIAL_NONE) {
            panels[n]->set_visibility(true);
        }
        else {
            RoomsEngine::switch_editor(SCENE_EDITOR);
        }
    });

    return new_panel;
}
