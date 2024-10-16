#include "inspector.h"

#include "framework/input.h"
#include "framework/ui/io.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"

#include "spdlog/spdlog.h"

namespace ui {

    uint32_t Inspector::row_id = 0;

    Inspector::Inspector(const InspectorDesc& desc, std::function<bool(Inspector*)> close_fn)
        : Node2D(name, desc.position, { 0.0f, 0.0f }), panel_size(desc.size), padding(desc.padding)
    {
        float inner_width = panel_size.x - padding * 2.0f;
        float inner_height = panel_size.y - padding * 2.0f;

        on_close = close_fn;

        root = new ui::XRPanel(name + "_background", panel_color, { 0.0f, 0.f }, panel_size);
        add_child(root);

        ui::VContainer2D* column = new ui::VContainer2D(name + "_column", glm::vec2(padding), colors::GREEN);
        column->set_fixed_size({ inner_width, inner_height });
        root->add_child(column);

        // Title
        float title_text_scale = 22.0f;
        float title_y_corrected = desc.title_height * 0.5f - title_text_scale * 0.5f;
        ui::Container2D* title_container = new ui::Container2D(name + "_title", { 0.0f, 0.0f }, { inner_width - padding * 0.4f, desc.title_height }, colors::BLUE);
        title_container->add_child(new ui::Text2D(desc.title.empty() ? "Inspector": desc.title, { 0.0f, title_y_corrected }, title_text_scale, ui::TEXT_CENTERED | ui::SKIP_TEXT_RECT));
        title_container->add_child(new ui::TextureButton2D("close_button", "data/textures/cross.png", ui::SKIP_NAME, { inner_width - padding * 3.0f, title_y_corrected }, glm::vec2(32.0f)));
        column->add_child(title_container);

        Node::bind("close_button", [&](const std::string& sg, void* data) {
            bool should_close = true;
            if (on_close) {
                should_close = on_close(this);
            }
            if (should_close) {
                set_visibility(false);
            }
        });

        // Body row
        body = new ui::VContainer2D(name + "_body", { 0.0f, 0.0f }, colors::RED);
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
            last_grab_position = { 0.0f, 0.0f };
            scroll_top = -glm::max(body_height - body->get_size().y, 0.0f);

            for (auto child : b->get_children()) {
                Node2D* row = static_cast<Node2D*>(child);
                row->translate({ 0.0f, scroll_top });
            }
        });
    }

    void Inspector::update(float delta_time)
    {
        sInputData data = root->get_input_data(true);

        if (data.is_hovered)
        {
            float scroll_dt = Input::get_mouse_wheel_delta() * 8.0f;

            if (data.was_pressed) {
                IO::set_focus(root);
            }

            if (data.is_pressed && !IO::is_focus_type(Node2DClassType::HSLIDER)) {
                scroll_dt += (last_grab_position.y - data.local_position.y);
            }

            if (data.was_released) {
                IO::set_focus(nullptr);
            }

            last_grab_position = data.local_position;

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
        }

        Node2D::update(delta_time);
    }

    void Inspector::clear()
    {
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

        IO::set_hover(nullptr, {});
    }

    void Inspector::label(const std::string& name, const std::string& text, uint32_t flags)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        flags |= (ui::SCROLLABLE | ui::TEXT_EVENTS | ui::DBL_CLICK | ui::LONG_CLICK);

        auto w = new ui::Text2D(text, 17.f, flags);
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

    void Inspector::button(const std::string& name, const std::string& texture_path, uint32_t flags)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        ui::Button2D* button = nullptr;

        if (flags & ui::CONFIRM_BUTTON) {
            button = new ui::ConfirmButton2D(name, texture_path, flags | ui::SKIP_NAME | ui::SCROLLABLE, { 0.0f, 0.0f }, glm::vec2(34.f));
        }
        else {
            button = new ui::TextureButton2D(name, texture_path, flags | ui::SKIP_NAME | ui::SCROLLABLE, { 0.0f, 0.0f }, glm::vec2(34.f));
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

        auto w = new ui::FloatSlider2D(name, "", value, { 0.0f, 0.0f }, glm::vec2(panel_size.x / 4.0f, 24.f), ui::SliderMode::HORIZONTAL, ui::SKIP_NAME | ui::SKIP_VALUE | ui::SCROLLABLE, min, max, precision);
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

        auto w = new ui::IntSlider2D(name, "", value, { 0.0f, 0.0f }, glm::vec2(panel_size.x / 4.0f, 24.f), ui::SliderMode::HORIZONTAL, ui::SKIP_NAME | ui::SKIP_VALUE | ui::SCROLLABLE, min, max);
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

    HContainer2D* Inspector::create_row()
    {
        ui::HContainer2D* new_row = new ui::HContainer2D("row_" + std::to_string(row_id++), { 0.0f, 0.0f });
        new_row->padding = glm::vec2(2.0f, 1.0f);
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
