#include "inspector.h"

#include "framework/input.h"
#include "framework/ui/io.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/nodes/viewport_3d.h"

#include "graphics/renderer.h"

#include "glm/gtx/quaternion.hpp"

#include "spdlog/spdlog.h"

namespace ui {

    uint32_t Inspector::row_id = 0;

    Inspector::Inspector(const InspectorDesc& desc)
        : Node2D(name, desc.position, { 0.0f, 0.0f }, ui::CREATE_3D), panel_size(desc.size), padding(desc.padding)
    {
        float inner_width = panel_size.x - padding * 2.0f;
        float inner_height = panel_size.y - padding * 2.0f;

        on_close = desc.close_fn;
        on_back = desc.back_fn;

        root = new ui::XRPanel(name + "_background", { 0.0f, 0.f }, panel_size, 0u, panel_color);
        add_child(root);

        ui::VContainer2D* column = new ui::VContainer2D(name + "_column", glm::vec2(padding));
        column->set_fixed_size({ inner_width, inner_height });
        root->add_child(column);

        // Title
        float title_text_scale = 22.0f;
        float title_y_corrected = desc.title_height * 0.5f - title_text_scale * 0.5f;
        ui::Container2D* title_container = new ui::Container2D(name + "_title", { 0.0f, 0.0f }, { inner_width - padding * 0.4f, desc.title_height });
        title = new ui::Text2D(desc.title.empty() ? "Inspector" : desc.title, { 0.0f, title_y_corrected }, title_text_scale, ui::TEXT_CENTERED | ui::SKIP_TEXT_RECT);
        back_button = new ui::TextureButton2D(name + "_back_button", { "data/textures/back.png", 0u, { padding, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Back" });
        close_button = new ui::TextureButton2D(name + "_close_button", { "data/textures/cross.png", 0u, { inner_width - padding * 3.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Close" });
        title_container->add_child(back_button);
        title_container->add_child(title);
        title_container->add_child(close_button);
        column->add_child(title_container);

        Node::bind(name + "_close_button", [&](const std::string& sg, void* data) {
            bool should_close = true;
            if (on_close) {
                should_close = on_close(this);
            }
            if (should_close) {
                set_visibility(false);
            }
        });

        Node::bind(name + "_back_button", [&](const std::string& sg, void* data) {
            if (on_back) {
                on_back(this);
            }
        });

        // Body row
        body = new ui::VContainer2D(name + "_body", { 0.0f, 0.0f });
        body->set_fixed_size({ inner_width, panel_size.y - desc.title_height - padding * 3.0f });
        column->add_child(body);

        // Listen for body children changes to update size
        Node::bind(name + "@children_changed", [&, b = body](const std::string& sg, void* data) {

            body_height = 0.0f;

            for (auto child : b->get_children()) {
                Node2D* row = static_cast<Node2D*>(child);
                body_height += row->get_size().y + 6.0f;
            }

            // Scroll to max_scroll
            last_scroll_position = { 0.0f, 0.0f };
            scroll_top = -glm::max(body_height - body->get_size().y, 0.0f);

            for (auto child : b->get_children()) {
                Node2D* row = static_cast<Node2D*>(child);
                row->translate({ 0.0f, scroll_top });
            }
        });
    }

    Inspector::~Inspector()
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

    void Inspector::update(float delta_time)
    {
        if ((IO::get_hover() == root) && Input::was_grab_pressed(HAND_RIGHT)) {
            grabbing = true;
        }

        if (Input::was_grab_released(HAND_RIGHT)) {
            grabbing = false;
        }

        root->set_priority(PANEL);

        auto renderer = Renderer::instance;

        if (renderer->get_xr_available()) {

            if (!placed) {
                glm::mat4x4 m(1.0f);
                glm::vec3 eye = renderer->get_camera_eye();
                glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.5f;

                m = glm::translate(m, new_pos);
                m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                set_xr_transform(Transform::mat4_to_transform(m));
                placed = true;
            }
            else if (grabbing) {

                Transform raycast_transform = Transform::mat4_to_transform(Input::get_controller_pose(HAND_RIGHT, POSE_AIM));
                const glm::vec3& forward = raycast_transform.get_front();

                glm::mat4x4 m(1.0f);
                glm::vec3 eye = raycast_transform.get_position();

                auto webgpu_context = Renderer::instance->get_webgpu_context();
                float width = static_cast<float>(webgpu_context->render_width);
                float height = static_cast<float>(webgpu_context->render_height);
                glm::vec2 grab_offset = glm::vec2(last_grab_position.x, panel_size.y - last_grab_position.y) / glm::vec2(width, height);
                glm::vec3 new_pos = eye + forward * last_grab_distance;

                m = glm::translate(m, new_pos);
                m = m * glm::toMat4(get_rotation_to_face(new_pos, renderer->get_camera_eye(), { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                m = glm::translate(m, -glm::vec3(grab_offset, 0.0f));
                set_xr_transform(Transform::mat4_to_transform(m));

                root->set_priority(DRAGGABLE);
            }
        }

        // Scroll stuff

        sInputData data = root->get_input_data(true);

        if (data.is_hovered)
        {
            float scroll_dt = Input::get_mouse_wheel_delta() * 8.0f;

            if (data.was_pressed) {
                IO::set_focus(root);
            }

            if (data.is_pressed && !IO::is_focus_type(Node2DClassType::HSLIDER)) {
                scroll_dt += (last_scroll_position.y - data.local_position.y);
            }

            if (data.was_released) {
                IO::set_focus(nullptr);
            }

            last_scroll_position = data.local_position;

            float max_scroll = glm::max(body_height - body->get_size().y, 0.0f);
            float new_scroll = scroll_top + scroll_dt;

            if (new_scroll > 0.0f) {
                scroll_dt = glm::abs(scroll_top);
            }
            else if (new_scroll < -max_scroll) {
                scroll_dt = -(max_scroll - glm::abs(scroll_top));
            }

            scroll_top += scroll_dt;

            // do the scroll..
            for (auto r : body->get_children()) {
                Node2D* row = static_cast<Node2D*>(r);
                row->translate({ 0.0f, scroll_dt });
            }

            if (renderer->get_xr_available() && grabbing && Input::was_grab_pressed(HAND_RIGHT)) {
                last_grab_position = last_scroll_position;
                last_grab_distance = data.ray_distance;
            }
        }

        Node2D::update(delta_time);
    }

    void Inspector::set_title(const std::string& new_title)
    {
        title->set_text(new_title);
    }

    void Inspector::clear_scroll()
    {
        last_scroll_position = {};

        // do the scroll..
        for (auto r : body->get_children()) {
            Node2D* row = static_cast<Node2D*>(r);
            row->translate({ 0.0f, -scroll_top });
        }

        scroll_top = 0.0f;
    }

    void Inspector::clear(uint8_t reset_flags, const std::string& new_title)
    {
        if (reset_flags & INSPECTOR_FLAG_FORCE_3D_POSITION) {
            placed = false;
        }

        // Show/Hide workflow buttons
        back_button->set_visibility(reset_flags & INSPECTOR_FLAG_BACK_BUTTON);
        close_button->set_visibility(reset_flags & INSPECTOR_FLAG_CLOSE_BUTTON);

        // Clear widgets inside the inspector

        std::vector<Node*> to_delete;

        for (auto node : body->get_children()) {
            to_delete.push_back(node);
        }

        std::function<void(Node*)> delete_node = [&](Node* node) {
            if (node == nullptr) return;
            while (!node->get_children().empty()) {
                auto child = node->get_children().back();
                delete_node(child);
            }
            delete node;
        };

        for (auto node : to_delete) {
            Node2D* node_2d = static_cast<Node2D*>(node);
            body->remove_child(node_2d);
            delete_node(node_2d);
        }

        items.clear();

        body_height = 0.0f;
        scroll_top = 0.0f;

        IO::set_hover(nullptr, {});

        if (new_title.size()) {
            set_title(new_title);
        }
    }

    void Inspector::label(const std::string& name, const std::string& text, uint32_t flags, const Color& c)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        flags |= (ui::SCROLLABLE | ui::DBL_CLICK | ui::LONG_CLICK | ui::TEXT_SELECTABLE);

        auto w = new ui::Text2D(text, { 0.0f, 0.0f }, 17.f, flags, c);
        w->set_signal(name);
        flex_container->add_child(w);
        items[name] = w;
    }

    void Inspector::icon(const std::string& texture_path)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        auto w = new ui::Image2D(texture_path, glm::vec2(36.f), ui::SCROLLABLE);
        flex_container->add_child(w);
        items[name] = w;
    }

    void Inspector::button(const std::string& name, const std::string& texture_path, uint32_t flags, const std::string& label)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        ui::Button2D* button = nullptr;

        sButtonDescription desc = { texture_path, flags | ui::SCROLLABLE, { 0.0f, 0.0f }, glm::vec2(34.f), colors::WHITE, label };

        if (flags & ui::CONFIRM_BUTTON) {
            button = new ui::ConfirmButton2D(name, desc);
        }
        else {
            button = new ui::TextureButton2D(name, desc);
        }

        flex_container->add_child(button);
        items[name] = button;
    }

    void Inspector::fslider(const std::string& name, float value, float* result, float min, float max, int precision)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        sSliderDescription desc = { .fvalue = value, .mode = ui::SliderMode::HORIZONTAL, .flags = ui::SKIP_NAME | ui::SKIP_VALUE | ui::SCROLLABLE,
            .fvalue_min = min, .fvalue_max = max, .precision = precision, .size = glm::vec2(panel_size.x / 4.0f, 24.f) };

        auto w = new ui::FloatSlider2D(name, desc);
        flex_container->add_child(w);
        items[name] = w;

        create_vector_component(flex_container, name, std::to_string(value), 'x', ui::SKIP_TEXT_RECT);

        Node::bind(name + "_x", (FuncFloat)[&](const std::string& signal, float value) {
            std::string new_value = std::to_string(value);
            Text2D* w = static_cast<Text2D*>(items[signal]);
            w->set_text(new_value.substr(0, new_value.find('.') + 3));
        });


        if (result != nullptr) {
            Node::bind(name, (FuncFloat)[result = result, n  = name](const std::string& signal, float value){
                *result = value;
                Node::emit_signal(n + "_x", value);
            });
        }
    }

    void Inspector::islider(const std::string& name, int value, int* result, int min, int max)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        sSliderDescription desc = { .ivalue = value, .mode = ui::SliderMode::HORIZONTAL, .flags = ui::SKIP_NAME | ui::SKIP_VALUE | ui::SCROLLABLE,
            .ivalue_min = min, .ivalue_max = max, .size = glm::vec2(panel_size.x / 4.0f, 24.f) };

        auto w = new ui::IntSlider2D(name, desc);
        flex_container->add_child(w);
        items[name] = w;

        create_vector_component(flex_container, name, std::to_string(value), 'x', ui::SKIP_TEXT_RECT);

        Node::bind(name + "_x", (FuncInt)[&](const std::string& signal, int value) {
            std::string new_value = std::to_string(value);
            Text2D* w = static_cast<Text2D*>(items[signal]);
            w->set_text(new_value.substr(0, new_value.find('.') + 3));
        });

        if (result != nullptr) {
            Node::bind(name, (FuncInt)[result = result](const std::string& signal, int value) {
                *result = value;
            });
        }
    }

    void Inspector::color_picker(const std::string& name, const Color& c, Color* result)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        float picker_size = PICKER_SIZE * 0.8f;
        auto w = new ui::ColorPicker2D(name, { 0.0f, 0.0f }, glm::vec2(picker_size), c, ui::SCROLLABLE);
        flex_container->add_child(w);
        // Translate after being added to the container
        w->translate({ (body->get_size().x - padding * 2.0f) * 0.5f - picker_size * 0.5f, 0.0f });
        items[name] = w;

        if (result != nullptr) {
            Node::bind(name, [result = result](const std::string& signal, Color value){
                *result = value;
            });
        }
    }

    void Inspector::create_vector_component(ui::HContainer2D* container, const std::string& name, const std::string& value, char component, uint32_t flags)
    {
        flags |= (ui::SCROLLABLE | ui::TEXT_EVENTS);
        std::string new_name = name + "_" + component;
        auto wx = new ui::Text2D(value.substr(0, value.find('.') + 3), 17.f, flags);
        wx->set_signal(new_name);
        container->add_child(wx);
        items[new_name] = wx;
    }

    void Inspector::same_line()
    {
        assert(!current_row && "Can't add in same line while in same line..!");
        current_row = create_row();
    }

    void Inspector::end_line()
    {
        current_row = nullptr;
    }

    void Inspector::set_indentation(uint32_t level)
    {
        indentation_level = level;
    }

    HContainer2D* Inspector::create_row()
    {
        ui::HContainer2D* new_row = new ui::HContainer2D("row_" + std::to_string(row_id++), { 0.0f, 0.0f });
        new_row->padding = glm::vec2(2.0f + indentation_level * 4.0f, 1.0f);
        new_row->item_margin = glm::vec2(4.0f, 0.0f);
        body->add_child(new_row);

        return new_row;
    }

    Node2D* Inspector::get(const std::string& name)
    {
        auto it = items.find(name);
        if (it == items.end())
            return nullptr;

        return (*it).second;
    }
}
