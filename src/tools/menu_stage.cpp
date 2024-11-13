#include "menu_stage.h"

#include "framework/nodes/panel_2d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"
#include "framework/ui/io.h"
#include "framework/math/math_utils.h"

#include "graphics/renderers/rooms_renderer.h"

#include "engine/rooms_engine.h"

#include "glm/gtx/quaternion.hpp"

void MenuStage::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    // Create tutorial/welcome panel
    {
        panel = new Node2D("tutorial_root", { 0.0f, 0.0f }, { 1.0f, 1.0f }, ui::CREATE_3D);

        generate_section("main_menu", "data/textures/tutorial/main_menu.png", MENU_SECTION_MAIN);
        generate_section("menu_discover", "data/textures/clone.png", MENU_SECTION_DISCOVER);

        current_panel = panels[MENU_SECTION_MAIN];
        current_panel->set_visibility(true);
    }
}

void MenuStage::update(float delta_time)
{
    current_panel->set_priority(PANEL);

    if ((IO::get_hover() == current_panel) && Input::was_grab_pressed(HAND_RIGHT)) {
        grabbing = true;
        last_grab_position = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    }

    if (Input::was_grab_released(HAND_RIGHT)) {
        grabbing = false;
    }

    if (renderer->get_openxr_available()) {

        if (!placed) {
            glm::mat4x4 m(1.0f);
            glm::vec3 eye = renderer->get_camera_eye();
            glm::vec3 new_pos = eye + renderer->get_camera_front() * 1.25f;

            m = glm::translate(m, new_pos);
            m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
            m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
            panel->set_xr_transform(Transform::mat4_to_transform(m));
        }

        if (grabbing) {

            Transform raycast_transform = Transform::mat4_to_transform(Input::get_controller_pose(HAND_RIGHT, POSE_AIM));
            const glm::vec3& forward = raycast_transform.get_front();

            glm::mat4x4 m(1.0f);
            glm::vec3 eye = raycast_transform.get_position();
            glm::vec3 delta_grab = (eye - last_grab_position) * 2.0f;
            float distance = glm::distance(eye - delta_grab, panel->get_xr_viewport()->get_translation());

            glm::vec3 new_pos = eye + forward * distance;

            m = glm::translate(m, new_pos);
            m = m * glm::toMat4(get_rotation_to_face(new_pos, renderer->get_camera_eye(), { 0.0f, 1.0f, 0.0f }));
            m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
            panel->set_xr_transform(Transform::mat4_to_transform(m));

            current_panel->set_priority(DRAGGABLE);
            placed = true;
            last_grab_position = eye;
        }
    }

    panel->update(delta_time);
}

void MenuStage::render()
{
    panel->render();
}

void MenuStage::generate_section(const std::string& name, const std::string& path, uint8_t section_idx)
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));
    glm::vec2 pos = { 0.0f, 0.0f };

    if (renderer->get_openxr_available()) {
        size = glm::vec2(1920.f, 1080.0f);
        pos = -size * 0.5f;
    }

    const glm::vec2& button_size = { size.x * 0.2f, size.y * 0.125f };
    const glm::vec2& mini_button_size = button_size * 0.75f;

    ui::XRPanel* new_panel = new ui::XRPanel(name, path, pos, size, ui::CURVED_PANEL | ui::FULLSCREEN);
    new_panel->set_visibility(false);
    panel->add_child(new_panel);

    switch (section_idx)
    {
    case MENU_SECTION_MAIN:
    {
        new_panel->add_button(name + "_discover", "data/textures/tutorial/discover_rooms.png", { size.x * 0.4f, size.y * 0.75f }, button_size);
        Node::bind(name + "_discover", [&, c = new_panel](const std::string& signal, void* button) {
            c->set_visibility(false);
            panels[MENU_SECTION_DISCOVER]->set_visibility(true);
            current_panel_idx = MENU_SECTION_DISCOVER;
            current_panel = panels[current_panel_idx];
        });

        new_panel->add_button(name + "_create", "data/textures/tutorial/create_room.png", { size.x * 0.6f, size.y * 0.75f }, button_size);
        Node::bind(name + "_create", [&](const std::string& signal, void* button) {
            RoomsEngine::switch_stage(SCENE_EDITOR);
        });
        break;
    }
    case MENU_SECTION_DISCOVER:
    {
        new_panel->add_button(name + "_back", "data/textures/tutorial/back.png", { size.x * 0.5f, size.y * 0.85f }, mini_button_size);
        Node::bind(name + "_back", [&, c = new_panel](const std::string& signal, void* button) {
            c->set_visibility(false);
            panels[MENU_SECTION_MAIN]->set_visibility(true);
            current_panel_idx = MENU_SECTION_MAIN;
            current_panel = panels[current_panel_idx];
        });
        break;
    }
    default:
        break;
    }

    panels[section_idx] = new_panel;
}
