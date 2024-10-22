#pragma once

#include "includes.h"

#include "framework/nodes/node.h"

#include <functional>

class Node2D;

namespace ui {

    class Text2D;
    class XRPanel;

    struct XrKey {
        std::string label = "";
        glm::vec2 position = { 0.0f, 0.0f };
        uint32_t flags = 0;
        std::string texture_path = "";
        glm::vec2 size = glm::vec2(42.0f);
    };

    struct XrKeyboardState {
        bool caps = false;
        bool caps_locked = false;
        bool layout_dirty = false;
        bool symbols = false;
        uint32_t p_caret = 1u;
        uint32_t max_length = 256u;
        std::string input = ":";
        ui::Text2D* text = nullptr;
        std::function<void(const std::string&)> callback;

        void set_input(const std::string& str);
        void clear_input();
        void push_char(char c);
        void remove_char();
        void reset();

        std::string get_input() { return input.substr(1); }
        float get_caret_position();

        void toggle_caps();
        void toggle_caps_lock();
        void toggle_symbols();
        void disable_caps();
    };

    class Keyboard {

        static glm::vec2 keyboard_size;
        static Node2D* keyboard_2d;
        static ui::Text2D* caret;

        static ui::XRPanel* root_common;
        static ui::XRPanel* root_lc;
        static ui::XRPanel* root_uc;
        static ui::XRPanel* root_sym;

        static float input_start_position;
        static bool active;

        static XrKeyboardState state;

        static void create_keyboard_common_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin);
        static void create_keyboard_letters_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin, bool upper_case = false);
        static void create_keyboard_symbols_layout(std::vector<XrKey>& keys, float start_x, float start_y, float margin);

        static void create_keys(float start_x, float start_y, float margin);
        static void bind_special_keys();

    public:

        Keyboard() {};

        static void initialize();

        static void render();
        static void update(float delta_time);

        static void request(std::function<void(const std::string&)> fn, const std::string& str = "", uint32_t max_length = 256);
        static void close() { active = false; };
    };
}
