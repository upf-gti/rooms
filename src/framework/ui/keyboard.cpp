#include "keyboard.h"

#include "framework/nodes/ui.h"
#include "framework/nodes/viewport_3d.h"

#include "graphics/renderer.h"

namespace ui {

    Node2D* Keyboard::keyboard_2d = nullptr;
    Viewport3D* Keyboard::xr_keyboard = nullptr;
    XrKeyboardState Keyboard::xr_keyboard_state;
    bool Keyboard::active = true;

    void XrKeyboardState::clear_input()
    {
        input = ":";
        text->set_text(input);
    }

    void XrKeyboardState::reset()
    {
        clear_input();
        caps = false;
    }

    void XrKeyboardState::push_char(char c)
    {
        input += c;
        text->set_text(input);
    };

    void XrKeyboardState::remove_char()
    {
        if (input.size() == 1) {
            // it should be the ":" char..
            return;
        }

        input.pop_back();
        text->set_text(input);
    }

    void XrKeyboardState::toggle_caps()
    {
        caps = !caps;

        if (!caps) {
            caps_locked = false;
        }
    }

    void XrKeyboardState::toggle_caps_lock()
    {
        if (!caps) {
            return;
        }

        caps_locked = !caps_locked;

        if (!caps_locked) {
            disable_caps();
        }
        else {
            caps = true;
        }
    }

    void XrKeyboardState::toggle_symbols()
    {
        symbols = !symbols;
    }

    void XrKeyboardState::disable_caps()
    {
        if (caps && !caps_locked) {
            Node::emit_signal("Shift@pressed", (void*)nullptr);
            caps = false;
        }
    }

    void Keyboard::initialize()
    {
        std::string name = "xr_keyboard";

        uint32_t max_col_count = 11;
        uint32_t max_row_count = 4;

        float input_height = 48.0f;
        float button_size = 42.0f;
        float button_margin = 5.0f;
        float full_size = button_size + button_margin;

        glm::vec2 start_pos = { full_size * 0.8f, full_size * 0.75f };
        glm::vec2 panel_size = glm::vec2(button_size * (max_col_count + 3u), button_size * (max_row_count + 2u) + input_height);
        Color panel_color = Color(0.0f, 0.0f, 0.0f, 0.8f);

        std::vector<XrKey> keys;
        create_keyboard_letters_layout(keys, start_pos.x, start_pos.y + input_height, button_margin);

        keyboard_2d = new Node2D(name, { 512.0f, 16.0f }, { 1.0f, 1.0f });

        ui::XRPanel* root = new ui::XRPanel(name + "@letters", panel_color, { 0.0f, 0.f }, panel_size);
        keyboard_2d->add_child(root);

        ui::XRPanel* root_symbols = new ui::XRPanel(name + "@symbols", panel_color, { 0.0f, 0.f }, panel_size);
        root_symbols->set_visibility(false);
        keyboard_2d->add_child(root_symbols);

        // Input text
        ui::Container2D* title_container = new ui::Container2D(name + "@title", { 0.0f, 0.0f }, { 512.0f, 48.f }, colors::BLUE);
        keyboard_2d->add_child(title_container);

        ui::Text2D* text = new ui::Text2D(":", { start_pos.x + 12.0f, start_pos.y + 8.0f }, 18.0f, ui::SKIP_TEXT_SHADOW);
        xr_keyboard_state.text = text;
        title_container->add_child(text);

        auto process_click = [rs = root_symbols, rl = root](const std::string& sg, void* data) {
            if (sg == "Shift") {
                xr_keyboard_state.toggle_caps();
            }
            else if (sg == "Backspace") {
                xr_keyboard_state.remove_char();
            }
            else if (sg == "Space") {
                xr_keyboard_state.push_char(' ');
            }
            else if (sg == "HideKeyboard") {
                xr_keyboard_state.reset();
                close();
            }
            else if (sg == "Enter") {
                // Send the input text where needed..
                // ...
                close();
            }
            else if (sg == "Symbols") {
                xr_keyboard_state.toggle_symbols();
                rs->set_visibility(xr_keyboard_state.symbols);
                rl->set_visibility(!xr_keyboard_state.symbols);
            }
            else {
                char c = sg[0];
                xr_keyboard_state.push_char(xr_keyboard_state.caps ? std::toupper(c) : c);
                xr_keyboard_state.disable_caps();
            }
        };

        auto process_dbl_click = [](const std::string& sg, void* data) {
            if (sg == "Shift@dbl_click") {
                xr_keyboard_state.toggle_caps_lock();
            }
        };

        for (const XrKey& key : keys) {
            root->add_child(new ui::TextureButton2D(key.label, key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size));

            if (key.bind_event) {
                Node::bind(key.label, process_click);
                Node::bind(key.label + "@dbl_click", process_dbl_click);
            }
        }

        // Fill symbols panel
        keys.clear();
        create_keyboard_symbols_layout(keys, start_pos.x, start_pos.y + input_height, button_margin);

        for (const XrKey& key : keys) {
            root_symbols->add_child(new ui::TextureButton2D(key.label, key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size));

            if (key.bind_event) {
                Node::bind(key.label, process_click);
            }
        }
    }

    void Keyboard::render()
    {
        if (Renderer::instance->get_openxr_available() && active) {
            xr_keyboard->render();
        }

        // debug for flat screen!! remove later
        keyboard_2d->render();
    }

    void Keyboard::update(float delta_time)
    {
        auto renderer = Renderer::instance;

        if (renderer->get_openxr_available() && active) {
            glm::mat4x4 m(1.0f);
            /*glm::vec3 eye = renderer->get_camera_eye();
            glm::vec3 new_pos = eye + renderer->get_camera_front() * 1.5f;

            m = glm::translate(m, new_pos);
            m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
            m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });*/

            xr_keyboard->set_model(m);
            xr_keyboard->update(delta_time);
        }

        // debug for flat screen!! remove later
        keyboard_2d->update(delta_time);
    }

    void Keyboard::create_keyboard_letters_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin)
    {
        glm::vec2 button_size = glm::vec2(42.0f);
        glm::vec2 spacing = button_size + margin;

        std::string first_row = "qwertyuiop";
        for (size_t i = 0; i < first_row.size(); ++i) {
            keys.push_back({ std::string(1, first_row[i]), glm::vec2(start_x + i * spacing.x, start_y) });
        }

        std::string second_row = "asdfghjkl";
        for (size_t i = 0; i < second_row.size(); ++i) {
            keys.push_back({ std::string(1, second_row[i]), glm::vec2(start_x + i * spacing.x + spacing.x * 0.10f, start_y + spacing.y) });
        }

        std::string third_row = "zxcvbnm,.";
        for (size_t i = 0; i < third_row.size(); ++i) {
            keys.push_back({ std::string(1, third_row[i]), glm::vec2(start_x + (i + 1) * spacing.x + spacing.x * 0.5f, start_y + spacing.y * 2.0f) });
        }

        // Additional rows (space, enter, backspace)

        keys.push_back({
            .label = "Backspace",
            .position = glm::vec2(start_x + first_row.size() * spacing.x, start_y),
            .texture_path = "data/textures/buttons/backspace_key.png"
        });

        keys.push_back({
            .label = "Shift",
            .position = glm::vec2(start_x + spacing.x * 0.5f, start_y + spacing.y * 2.0f),
            .flags = ui::ALLOW_TOGGLE | ui::DBL_CLICK,
            .texture_path = "data/textures/buttons/shift_key.png"
        });

        keys.push_back({
            .label = "Enter",
            .position = glm::vec2(start_x + second_row.size() * spacing.x + spacing.x * 0.10f, start_y + spacing.y),
            .size = { button_size.x * 2.0f, button_size.y }
        });

        start_x += 42.f;

        keys.push_back({
            .label = "Symbols",
            .position = glm::vec2(start_x + spacing.x, start_y + spacing.y * 3.0f),
            .flags = ui::ALLOW_TOGGLE,
            .texture_path = "data/textures/buttons/symbols_key.png"
        });

        float space_key_width = spacing.x * 5.0f;

        keys.push_back({
            .label = "Space",
            .position = glm::vec2(start_x + spacing.x * 2.0f, start_y + spacing.y * 3.0f),
            .size = { space_key_width, button_size.y }
        });

        keys.push_back({
            .label = "HideKeyboard",
            .position = glm::vec2(start_x + spacing.x * 2.0f + space_key_width + margin, start_y + spacing.y * 3.0f),
            .texture_path = "data/textures/buttons/hide_keyboard_key.png"
        });
    }

    void Keyboard::create_keyboard_symbols_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin)
    {
        glm::vec2 button_size = glm::vec2(42.0f);
        glm::vec2 spacing = button_size + margin;

        std::string first_row = "0123456789";
        for (size_t i = 0; i < first_row.size(); ++i) {
            keys.push_back({ std::string(1, first_row[i]), glm::vec2(start_x + i * spacing.x, start_y) });
        }

        std::string second_row = "-_/:;()&@";
        for (size_t i = 0; i < second_row.size(); ++i) {
            keys.push_back({ std::string(1, second_row[i]), glm::vec2(start_x + i * spacing.x + spacing.x * 0.10f, start_y + spacing.y) });
        }

        std::string third_row = "?!$<>%+*=";
        for (size_t i = 0; i < third_row.size(); ++i) {
            keys.push_back({ std::string(1, third_row[i]), glm::vec2(start_x + (i + 1) * spacing.x + spacing.x * 0.5f, start_y + spacing.y * 2.0f) });
        }

        // Additional rows (space, enter, backspace)
        // This buttons will be duplicated, but they won't be using any callback, it's for the sake of simplicity

        keys.push_back({
            .label = "Backspace",
            .position = glm::vec2(start_x + first_row.size() * spacing.x, start_y),
            .texture_path = "data/textures/buttons/backspace_key.png",
            .bind_event = false
        });

        keys.push_back({
            .label = "Shift",
            .position = glm::vec2(start_x + spacing.x * 0.5f, start_y + spacing.y * 2.0f),
            .flags = ui::DISABLED,
            .texture_path = "data/textures/buttons/shift_key.png",
            .bind_event = false
        });

        keys.push_back({
            .label = "Enter",
            .position = glm::vec2(start_x + second_row.size() * spacing.x + spacing.x * 0.10f, start_y + spacing.y),
            .size = { button_size.x * 2.0f, button_size.y },
            .bind_event = false
        });

        start_x += 42.f;

        keys.push_back({
            .label = "Symbols",
            .position = glm::vec2(start_x + spacing.x, start_y + spacing.y * 3.0f),
            .flags = ui::ALLOW_TOGGLE,
            .texture_path = "data/textures/buttons/symbols_key.png",
            .bind_event = false
        });

        float space_key_width = spacing.x * 5.0f;

        keys.push_back({
            .label = "Space",
            .position = glm::vec2(start_x + spacing.x * 2.0f, start_y + spacing.y * 3.0f),
            .size = { space_key_width, button_size.y },
            .bind_event = false
        });

        keys.push_back({
            .label = "HideKeyboard",
            .position = glm::vec2(start_x + spacing.x * 2.0f + space_key_width + margin, start_y + spacing.y * 3.0f),
            .texture_path = "data/textures/buttons/hide_keyboard_key.png",
            .bind_event = false
        });
    }
}
