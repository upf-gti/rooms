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

        // Welcome panel
        {
            welcome_panel = new ui::XRPanel("scene_editor_root", "data/images/welcome_screen.png", pos, size);
            xr_panel_2d->add_child(welcome_panel);

            welcome_panel->add_button("start_tutorial", "data/textures/menu_buttons/start_tutorial.png", { button_size.x * 0.75f, button_size.y }, button_size);
            welcome_panel->add_button("create_button", "data/textures/menu_buttons/create_now.png", { size.x - button_size.x * 0.75f, button_size.y }, button_size);
        }

        // Rooms intro panel
        {
            rooms_intro_panel = new ui::XRPanel("scene_editor_root", "data/images/rooms_intro.png", pos, size);
            rooms_intro_panel->set_visibility(false);
            xr_panel_2d->add_child(rooms_intro_panel);

            rooms_intro_panel->add_button("skip_button", "data/textures/menu_buttons/next.png", { size.x * 0.5f, button_size.y }, button_size);
        }

        if (renderer->get_openxr_available()) {
            xr_panel_3d = new Viewport3D(xr_panel_2d);
            xr_panel_3d->set_active(true);
        }
    }

    // Bind button events
    {
        Node::bind("create_button", [&](const std::string& signal, void* button) {
            RoomsEngine::switch_editor(SCENE_EDITOR);
        });

        Node::bind("start_tutorial", [&](const std::string& signal, void* button) {
            welcome_panel->set_visibility(false);
            rooms_intro_panel->set_visibility(true);
        });
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
        glm::vec3 new_pos = eye + renderer->get_camera_front();

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));

        xr_panel_3d->set_model(m);
        xr_panel_3d->update(delta_time);
    }
    else {
        xr_panel_2d->update(delta_time);
    }
}

void TutorialEditor::render()
{
    RoomsEngine::render_controllers();

    if (xr_panel_3d) {
        xr_panel_3d->render();
    }
    else {
        xr_panel_2d->render();
    }
}
