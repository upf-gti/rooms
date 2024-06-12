#include "inspector.h"

#include "framework/input.h"
#include "framework/ui/io.h"

namespace ui {

    uint32_t Inspector::row_id = 0;

    Inspector::Inspector(const InspectorDesc& desc)
        : Node2D(name, desc.position, { 0.0f, 0.0f }), panel_size(desc.size), padding(desc.padding)
    {
        float inner_width = panel_size.x - padding * 2.0f;
        float inner_height = panel_size.y - padding * 2.0f;

        root = new ui::XRPanel(name + "@background", panel_color, { 0.0f, 0.f }, panel_size);
        add_child(root);

        ui::VContainer2D* column = new ui::VContainer2D(name + "@column", glm::vec2(padding), colors::GREEN);
        column->set_fixed_size({ inner_width, inner_height });
        root->add_child(column);

        // Title
        float title_text_scale = 22.0f;
        float title_y_corrected = desc.title_height * 0.5f - title_text_scale * 0.5f;
        ui::Container2D* title_container = new ui::Container2D(name + "@title", { 0.0f, 0.0f }, { inner_width - padding * 0.4f, desc.title_height }, colors::BLUE);
        title_container->add_child(new ui::Text2D("Inspector", { 0.0f, title_y_corrected }, title_text_scale, ui::TEXT_CENTERED));
        column->add_child(title_container);

        // Body row
        body = new ui::VContainer2D(name + "@body", { 0.0f, 0.0f }, colors::RED);
        body->set_fixed_size({ inner_width, panel_size.y - desc.title_height - padding * 4.0f });
        column->add_child(body);
    }

    void Inspector::update(float delta_time)
    {
        sInputData data = root->get_input_data(true);

        if (data.is_hovered)
        {
            float scroll_dt = Input::get_mouse_wheel_delta() * 8.0f;

            if (data.was_pressed) {
                grabbing = true;
            }

            if (data.is_pressed) {
                grabbing = true;
                scroll_dt += (last_grab_position.y - data.local_position.y);
            }

            if (data.was_released) {
                grabbing = false;
            }

            last_grab_position = data.local_position;

            float new_scroll = scroll_top + scroll_dt;
            float max_scroll = body->get_size().y * 0.5f;
            if (new_scroll > 0.0f) {
                scroll_dt = glm::abs(scroll_top);
            }
            else if (new_scroll < -max_scroll) {
                scroll_dt = max_scroll - glm::abs(scroll_top);
            }

            scroll_top += scroll_dt;

            // do the scroll..
            for (auto r : body->get_children())
            {
                Node2D* row = static_cast<Node2D*>(r);
                row->translate({ 0.0f, scroll_dt });
            }
        }

        Node2D::update(delta_time);
    }


    void Inspector::add_label(const std::string& label)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        flex_container->add_child(new ui::Text2D(label, 16.f, ui::SCROLLABLE));
    }

    void Inspector::add_button(const std::string& name, const std::string& texture_path, uint32_t flags)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        flex_container->add_child(new ui::TextureButton2D(name, texture_path, flags | ui::SKIP_NAME | ui::SCROLLABLE, { 0.0f, 0.0f }, glm::vec2(32.f)));
    }

    void Inspector::add_slider(const std::string& name, float value, float min, float max, int precision)
    {
        ui::HContainer2D* flex_container = current_row;

        if (!flex_container) {
            flex_container = create_row();
        }

        flex_container->add_child(new ui::Slider2D(name, "", value, { 0.0f, 0.0f }, glm::vec2(panel_size.x / 4.0f, 24.f), ui::SliderMode::HORIZONTAL, ui::SKIP_NAME | ui::SKIP_VALUE | ui::SCROLLABLE, min, max, precision));
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
}
