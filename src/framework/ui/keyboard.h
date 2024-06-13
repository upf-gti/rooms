#pragma once

#include "includes.h"

#include "framework/nodes/node.h"

class Node2D;
class Viewport3D;

namespace ui {

    class Text2D;

    struct XrKey {
        std::string label = "";
        glm::vec2 position = { 0.0f, 0.0f };
        uint32_t flags = 0;
        std::string texture_path = "";
        glm::vec2 size = glm::vec2(42.0f);
        bool bind_event = true;
    };

    struct XrKeyboardState {
        bool caps = false;
        bool caps_locked = false;
        bool symbols = false;
        std::string input = ":";
        ui::Text2D* text = nullptr;

        const std::string& get_input() { return input; }
        void clear_input();
        void push_char(char c);
        void remove_char();
        void reset();

        void toggle_caps();
        void toggle_caps_lock();
        void toggle_symbols();
        void disable_caps();
    };

    class Keyboard {

        static Node2D* keyboard_2d;
        static Viewport3D* xr_keyboard;
        static bool active;

        static XrKeyboardState xr_keyboard_state;

        static void create_keyboard_letters_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin);
        static void create_keyboard_symbols_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin);

    public:

        Keyboard() {};

        static void initialize();

        static void render();
        static void update(float delta_time);

        static void request() { active = true; };
        static void close() { active = false; };
    };
}
