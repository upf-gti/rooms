#include "context_menu.h"

#include "framework/input.h"
#include "framework/ui/io.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/nodes/viewport_3d.h"

#include "graphics/renderer.h"

#include "engine/rooms_engine.h"

#include "glm/gtx/quaternion.hpp"

#include "spdlog/spdlog.h"

namespace ui {

    uint32_t ContextMenu::row_id = 0;

    ContextMenu::ContextMenu(const glm::vec2& position, const std::vector<sContextMenuOption>& new_options)
        : Node2D(name, { 0.0f, 0.0f }, { 0.0f, 0.0f }, ui::CREATE_3D)
    {
        glm::vec2 body_padding = { 12.0f, 16.0f };
        float option_margin_y = 4.0f;

        panel_size = { 200.f, new_options.size() * 24.f + (new_options.size() - 1u) * option_margin_y };

        glm::vec2 real_size = panel_size + body_padding * 2.0f;

        glm::vec2 centered_position = position;
        centered_position -= real_size * 0.5f;
        set_position(centered_position);

        root = new ui::XRPanel(name + "_background", { 0.0f, 0.0f }, real_size, 0u, panel_color);
        add_child(root);

        // Body row
        body = new ui::VContainer2D(name + "_body", { 0.0f, 0.0f });
        body->item_margin = { 0.0f, option_margin_y };
        body->padding = body_padding;
        body->set_fixed_size(real_size);
        root->add_child(body);

        // Set xr position
        auto renderer = Renderer::instance;

        if (renderer->get_openxr_available()) {
            glm::mat4x4 m(1.0f);
            glm::vec3 eye = renderer->get_camera_eye();
            glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.5f;

            m = glm::translate(m, new_pos);
            m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
            m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
            set_xr_transform(Transform::mat4_to_transform(m));
        }

        for (const auto& option : new_options) {
            add_option(option);
        }

        static_cast<RoomsEngine*>(RoomsEngine::get_instance())->push_context_menu(this);
    }

    ContextMenu::~ContextMenu()
    {
        std::function<void(Node* node)> fn_delete = [&](Node* node) {

            if (!node) return;

            while (node->get_children().size()) {
                fn_delete(node->get_children().back());
            }

            delete node;
        };

        fn_delete(root);
    }

    void ContextMenu::update(float delta_time)
    {
        Node2D::update(delta_time);

        // delete if unhovered..

        sInputData data = root->get_input_data(true);

        if (!data.is_hovered) {
            static_cast<RoomsEngine*>(RoomsEngine::get_instance())->delete_context_menu(this);
        }
    }

    void ContextMenu::add_option(const sContextMenuOption& option)
    {
        ui::HContainer2D* flex_container = create_row();

        sButtonDescription desc = { "", ui::SKIP_HOVER_SCALE, { 0.0f, 0.0f }, glm::vec2(panel_size.x, 24.0f) };

        ui::Button2D* button = new ui::TextureButton2D(option.name, desc);

        Node::bind(option.name, [scope = this, o = option, index = option_count](const std::string& signal, void* data) {
            if (std::holds_alternative<bool*>(o.event)) {
                bool* var = std::get<bool*>(o.event);
                *var = !(*var);
            }
            else if (std::holds_alternative<FuncInt>(o.event)) {
                FuncInt var = std::get<FuncInt>(o.event);
                var(signal, index);
            }

            static_cast<RoomsEngine*>(RoomsEngine::get_instance())->delete_context_menu(scope);
        });

        flex_container->add_child(button);

        option_count++;
    }

    HContainer2D* ContextMenu::create_row()
    {
        ui::HContainer2D* new_row = new ui::HContainer2D("row_" + std::to_string(row_id++), { 0.0f, 0.0f });
        new_row->padding = glm::vec2(0.0f);
        new_row->item_margin = glm::vec2(0.0f);
        body->add_child(new_row);

        return new_row;
    }
}
