#include "keyboard.h"

#include "framework/nodes/panel_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"

#include "framework/math/math_utils.h"

#include "graphics/renderers/rooms_renderer.h"

#include "glm/gtx/quaternion.hpp"

namespace ui {

    Node2D* Keyboard::keyboard_2d = nullptr;
    XrKeyboardState Keyboard::state;
    glm::vec2 Keyboard::keyboard_size = {};
    ui::Text2D* Keyboard::caret = nullptr;

    bool Keyboard::active = false;
    bool Keyboard::placed = false;
    float Keyboard::input_start_position = 0.0f;

    ui::XRPanel* Keyboard::root_common = nullptr;
    ui::XRPanel* Keyboard::root_lc = nullptr;
    ui::XRPanel* Keyboard::root_uc = nullptr;
    ui::XRPanel* Keyboard::root_sym = nullptr;

    const uint16_t MAX_PHYSICAL_KEYS = 40;
    const uint16_t physical_keys[MAX_PHYSICAL_KEYS] = {
        GLFW_KEY_SPACE, GLFW_KEY_COMMA, GLFW_KEY_MINUS, GLFW_KEY_PERIOD,
        GLFW_KEY_0, GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4, GLFW_KEY_5, GLFW_KEY_6, GLFW_KEY_7, GLFW_KEY_8, GLFW_KEY_9,
        GLFW_KEY_A, GLFW_KEY_B, GLFW_KEY_C, GLFW_KEY_D, GLFW_KEY_E, GLFW_KEY_F, GLFW_KEY_G, GLFW_KEY_H, GLFW_KEY_I, GLFW_KEY_J,
        GLFW_KEY_K, GLFW_KEY_L, GLFW_KEY_M, GLFW_KEY_N, GLFW_KEY_O, GLFW_KEY_P, GLFW_KEY_Q, GLFW_KEY_R, GLFW_KEY_S, GLFW_KEY_T,
        GLFW_KEY_U, GLFW_KEY_V, GLFW_KEY_W, GLFW_KEY_X, GLFW_KEY_Y, GLFW_KEY_Z
    };

    void XrKeyboardState::set_input(const std::string& str)
    {
        input = ":" + str.substr(0, max_length);
        text->set_text(input);
        p_caret = input.size();
    }

    void XrKeyboardState::clear_input()
    {
        input = ":";
        text->set_text(input);
        p_caret = 1u;
    }

    void XrKeyboardState::reset()
    {
        clear_input();
        caps = false;
        symbols = false;
        caps_locked = false;
    }

    void XrKeyboardState::push_char(char c)
    {
        if (input.size() >= max_length) {
            return;
        }

        bool use_caps = caps || Input::is_grab_pressed(HAND_RIGHT);
        input += (use_caps ? std::toupper(c) : c);
        text->set_text(input);
        p_caret++;
        disable_caps();
    };

    void XrKeyboardState::remove_char()
    {
        if (input.size() == 1) {
            // it should be the ":" char..
            return;
        }

        input.erase(input.begin() + p_caret - 1u);
        text->set_text(input);
        p_caret--;
    }

    void XrKeyboardState::toggle_caps()
    {
        caps = !caps;

        if (!caps) {
            caps_locked = false;
        }

        layout_dirty = true;
    }

    void XrKeyboardState::toggle_caps_lock()
    {
        // since it's double click, the first click toggled it..
        //bool is_caps = !caps;

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

        layout_dirty = true;
    }

    void XrKeyboardState::toggle_symbols()
    {
        symbols = !symbols;
        layout_dirty = true;
    }

    void XrKeyboardState::disable_caps()
    {
        if (caps && !caps_locked) {
            Node::emit_signal("Shift@pressed", (void*)nullptr);
            caps = false;
            layout_dirty = true;
        }
    }

    void XrKeyboardState::move_caret_left()
    {
        p_caret--;
        p_caret = glm::max((uint8_t)1u, p_caret);
    }

    void XrKeyboardState::move_caret_right()
    {
        p_caret++;
        p_caret = glm::min(p_caret, (uint8_t)input.size());
    }

    float XrKeyboardState::get_caret_position()
    {
        return text->text_entity->get_text_width(input.substr(0, p_caret));
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
        keyboard_size = glm::vec2(button_size * (max_col_count + 3u), button_size * (max_row_count + 2u) + input_height);
        Color panel_color = Color(0.0f, 0.0f, 0.0f, 0.8f);

        keyboard_2d = new Node2D(name, { 0.0f, 0.0f }, { 1.0f, 1.0f }, ui::CREATE_3D);

        root_common = new ui::XRPanel(name + "_common", panel_color, { 0.0f, 0.f }, keyboard_size);
        root_common->render_background = false;
        keyboard_2d->add_child(root_common);

        root_lc = new ui::XRPanel(name + "_letters", panel_color, { 0.0f, 0.f }, keyboard_size);
        keyboard_2d->add_child(root_lc);

        root_uc = new ui::XRPanel(name + "_uppercase", panel_color, { 0.0f, 0.f }, keyboard_size);
        root_uc->set_visibility(false);
        keyboard_2d->add_child(root_uc);

        root_sym = new ui::XRPanel(name + "_symbols", panel_color, { 0.0f, 0.f }, keyboard_size);
        root_sym->set_visibility(false);
        keyboard_2d->add_child(root_sym);

        // Input text
        ui::Container2D* title_container = new ui::Container2D(name + "_title", { 0.0f, 0.0f }, { 512.0f, 48.f });
        keyboard_2d->add_child(title_container);

        input_start_position = start_pos.x + 12.0f;
        state.text = new ui::Text2D(":", { input_start_position, start_pos.y + 8.0f }, 18.0f, ui::SKIP_TEXT_RECT);
        title_container->add_child(state.text);

        // Create caret (will blink and be moved at the desired position)
        caret = new ui::Text2D("|", { input_start_position, start_pos.y + 8.0f }, 18.0f, ui::SKIP_TEXT_RECT);
        title_container->add_child(caret);

        // Crete and bind events for each key
        create_keys(start_pos.x, start_pos.y + input_height, button_margin);

        bind_special_keys();

        // Move keyboard to center it (at least in flat screen)
        auto webgpu_context = Renderer::instance->get_webgpu_context();
        glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

        keyboard_2d->translate({ screen_size.x * 0.5f - keyboard_size.x * 0.5f, 16.0f});
    }

    void Keyboard::render()
    {
        if (!active) {
            return;
        }

        keyboard_2d->render();
    }

    void Keyboard::update(float delta_time)
    {
        if (!active) {
            return;
        }

        auto renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

        // Set layout
        if(state.layout_dirty) {
            root_lc->set_visibility(!state.symbols && !state.caps);
            root_uc->set_visibility(!state.symbols && state.caps);
            root_sym->set_visibility(state.symbols);
            state.layout_dirty = false;
        }

        // move and blink caret
        {
            static float timer = 0.0f;
            timer += delta_time;
            float x_pos = input_start_position + state.get_caret_position();
            caret->set_position({ x_pos - 2.0f, caret->get_local_translation().y });
            if (timer > 0.5f) {
                caret->set_visibility(!caret->get_visibility());
                timer = 0.0f;
            }
        }

        if (renderer->get_openxr_available()) {

            if (!placed) {
                glm::mat4x4 m(1.0f);
                glm::vec3 eye = renderer->get_camera_eye();
                glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.5f;

                // Set current pos
                m = glm::translate(m, new_pos);
                // Rotate to face camera
                m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                // Center panel
                glm::vec2 offset = -(keyboard_size * 0.5f * keyboard_2d->get_scale());
                m = glm::translate(m, glm::vec3(offset.x, -offset.y, 0.0f));

                keyboard_2d->set_xr_transform(Transform::mat4_to_transform(m));
                placed = true;
            }
            // else stay in position..
        }
        else {
            // Push keys in 2d using physical keyboard..
            for (uint32_t i = 0u; i < MAX_PHYSICAL_KEYS; ++i) {
                if (Input::was_key_pressed(physical_keys[i], true)) {
                    char c = physical_keys[i];
                    state.push_char(Input::is_key_pressed(GLFW_KEY_LEFT_SHIFT, true) ? c : std::tolower(c));
                }
            }

            if (Input::was_key_pressed(GLFW_KEY_ENTER, true)) { Node::emit_signal("Enter", (void*)nullptr); }
            if (Input::was_key_pressed(GLFW_KEY_BACKSPACE, true)) { Node::emit_signal("Backspace", (void*)nullptr); }
            if (Input::was_key_pressed(GLFW_KEY_ESCAPE, true)) { Node::emit_signal("HideKeyboard", (void*)nullptr); }

            if (Input::was_key_pressed(GLFW_KEY_LEFT, true)) { state.move_caret_left(); }
            if (Input::was_key_pressed(GLFW_KEY_RIGHT, true)) { state.move_caret_right(); }
        }

        keyboard_2d->update(delta_time);
    }

    void Keyboard::request(std::function<void(const std::string&)> fn, const std::string& str, uint8_t max_length)
    {
        active = true;
        placed = false; // place again in correct position

        state.callback = fn;
        state.max_length = max_length;

        if (str.size() > 0u) {
            state.set_input(str);
        }
    }

    void Keyboard::create_keys(float start_x, float start_y, float margin)
    {
        // Create common keys panel
        std::vector<XrKey> keys;
        create_keyboard_common_layout(keys, start_x, start_y, margin);

        for (const XrKey& key : keys) {
            root_common->add_child(new ui::TextureButton2D(key.label, { key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size }));
        }

        // Create lower case letter panel
        keys.clear();
        create_keyboard_letters_layout(keys, start_x, start_y, margin);

        for (const XrKey& key : keys) {
            root_lc->add_child(new ui::TextureButton2D(key.label, { key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size }));
        }

        // Fill upper case letters panel
        keys.clear();
        create_keyboard_letters_layout(keys, start_x, start_y, margin, true);

        for (const XrKey& key : keys) {
            root_uc->add_child(new ui::TextureButton2D(key.label, { key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size }));
        }

        // Fill symbols panel
        keys.clear();
        create_keyboard_symbols_layout(keys, start_x, start_y, margin);

        for (const XrKey& key : keys) {
            root_sym->add_child(new ui::TextureButton2D(key.label, { key.texture_path, key.flags | ui::SKIP_NAME, key.position, key.size }));
        }
    }

    void Keyboard::bind_special_keys()
    {
        // Backspace
        {
            Node::bind("Backspace", [&](const std::string& sg, void* data) {
                state.remove_char();
            });

            Node::bind("Backspace@long_click", [&](const std::string& sg, void* data) {
                state.clear_input();
            });
        }

        // Shift
        {
            Node::bind("Shift", [&](const std::string& sg, void* data) {
                state.toggle_caps();
            });

            Node::bind("Shift@dbl_click", [&](const std::string& sg, void* data) {
                state.toggle_caps_lock();
            });
        }

        // Enter
        Node::bind("Enter", [&](const std::string& sg, void* data) {
            const std::string& str = state.get_input();
            if (str.size()) {
                state.callback(str);
                state.reset();
                close();
            }
        });

        // Symbols key
        Node::bind("Symbols", [&](const std::string& sg, void* data) {
            state.toggle_symbols();
        });

        // Space
        Node::bind("Space", [&](const std::string& sg, void* data) {
            state.push_char(' ');
        });

        // Hide Keyboard key
        Node::bind("HideKeyboard", [&](const std::string& sg, void* data) {
            state.reset();
            close();
        });
    }

    void Keyboard::create_keyboard_common_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin)
    {
        glm::vec2 button_size = glm::vec2(42.0f);
        glm::vec2 spacing = button_size + margin;

        size_t first_row_size = 10;
        size_t second_row_size = 9;

        // Backspace
        keys.push_back({
            .label = "Backspace",
            .position = glm::vec2(start_x + first_row_size * spacing.x, start_y),
            .flags = ui::LONG_CLICK,
            .texture_path = "data/textures/buttons/backspace_key.png"
        });

        // Shift
        keys.push_back({
            .label = "Shift",
            .position = glm::vec2(start_x + spacing.x * 0.5f, start_y + spacing.y * 2.0f),
            .flags = ui::ALLOW_TOGGLE | ui::DBL_CLICK,
            .texture_path = "data/textures/buttons/shift_key.png"
        });

        // Enter
        keys.push_back({
            .label = "Enter",
            .position = glm::vec2(start_x + second_row_size * spacing.x + spacing.x * 0.10f, start_y + spacing.y),
            .size = { button_size.x * 2.0f, button_size.y }
        });

        start_x += 42.f;

        // Symbols key
        keys.push_back({
            .label = "Symbols",
            .position = glm::vec2(start_x + spacing.x, start_y + spacing.y * 3.0f),
            .flags = ui::ALLOW_TOGGLE,
            .texture_path = "data/textures/buttons/symbols_key.png"
        });

        float space_key_width = spacing.x * 5.0f;

        // Space
        keys.push_back({
            .label = "Space",
            .position = glm::vec2(start_x + spacing.x * 2.0f, start_y + spacing.y * 3.0f),
            .size = { space_key_width, button_size.y }
        });

        // Hide Keyboard key
        keys.push_back({
            .label = "HideKeyboard",
            .position = glm::vec2(start_x + spacing.x * 2.0f + space_key_width + margin, start_y + spacing.y * 3.0f),
            .texture_path = "data/textures/buttons/hide_keyboard_key.png"
        });
    }

    void Keyboard::create_keyboard_letters_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin, bool upper_case)
    {
        glm::vec2 button_size = glm::vec2(42.0f);
        glm::vec2 spacing = button_size + margin;

        auto fn_char = [&](const std::string& sg, void* data) {
            state.push_char(sg[0]);
        };

        std::string first_row = upper_case ? "QWERTYUIOP" : "qwertyuiop";
        for (size_t i = 0; i < first_row.size(); ++i) {
            const std::string& label = std::string(1, first_row[i]);
            keys.push_back({ label, glm::vec2(start_x + i * spacing.x, start_y) });
            Node::bind(label, fn_char);
        }

        std::string second_row = upper_case ? "ASDFGHJKL" : "asdfghjkl";
        for (size_t i = 0; i < second_row.size(); ++i) {
            const std::string& label = std::string(1, second_row[i]);
            keys.push_back({ label, glm::vec2(start_x + i * spacing.x + spacing.x * 0.10f, start_y + spacing.y) });
            Node::bind(label, fn_char);
        }

        std::string third_row = upper_case ? "ZXCVBNM,." : "zxcvbnm,.";
        for (size_t i = 0; i < third_row.size(); ++i) {
            const std::string& label = std::string(1, third_row[i]);
            keys.push_back({ label, glm::vec2(start_x + (i + 1) * spacing.x + spacing.x * 0.5f, start_y + spacing.y * 2.0f) });
            Node::bind(label, fn_char);
        }
    }

    void Keyboard::create_keyboard_symbols_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin)
    {
        glm::vec2 button_size = glm::vec2(42.0f);
        glm::vec2 spacing = button_size + margin;

        auto fn_char = [&](const std::string& sg, void* data) {
            state.push_char(sg[0]);
        };

        std::string first_row = "0123456789";
        for (size_t i = 0; i < first_row.size(); ++i) {
            const std::string& label = std::string(1, first_row[i]);
            keys.push_back({ label, glm::vec2(start_x + i * spacing.x, start_y) });
            Node::bind(label, fn_char);
        }

        std::string second_row = "-_/:;()&@";
        for (size_t i = 0; i < second_row.size(); ++i) {
            const std::string& label = std::string(1, second_row[i]);
            keys.push_back({ label, glm::vec2(start_x + i * spacing.x + spacing.x * 0.10f, start_y + spacing.y) });
            Node::bind(label, fn_char);
        }

        std::string third_row = "?!$<>%+*=";
        for (size_t i = 0; i < third_row.size(); ++i) {
            const std::string& label = std::string(1, third_row[i]);
            keys.push_back({ label, glm::vec2(start_x + (i + 1) * spacing.x + spacing.x * 0.5f, start_y + spacing.y * 2.0f) });
            Node::bind(label, fn_char);
        }
    }
}
