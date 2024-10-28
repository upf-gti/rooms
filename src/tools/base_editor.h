#pragma once

#include <string>
#include <unordered_map>

#include "framework/input_xr.h"

class RoomsRenderer;
class Node2D;

namespace ui {
    class HContainer2D;
    class VContainer2D;
}

namespace shortcuts {
    // Left hand
    const std::string X_BUTTON_PATH = "data/textures/buttons/x.png";
    const std::string Y_BUTTON_PATH = "data/textures/buttons/y.png";
    const std::string L_TRIGGER_PATH = "data/textures/buttons/l_trigger.png";

    // Right hand
    const std::string A_BUTTON_PATH = "data/textures/buttons/a.png";
    const std::string B_BUTTON_PATH = "data/textures/buttons/b.png";
    const std::string R_TRIGGER_PATH = "data/textures/buttons/r_trigger.png";
}

class BaseEditor {

    std::string name;

protected:

    bool is_shift_left_pressed = false;
    bool is_shift_right_pressed = false;

    float last_hover_time = 0.0f;

    struct sControllerMovementData {
        glm::vec3 prev_position = {};
        glm::vec3 velocity = {};
        glm::vec3 acceleration = {};
        glm::vec3 frame_distance = {};
        glm::vec3 prev_edit_position = {};
    };

    sControllerMovementData controller_movement_data[HAND_COUNT];

    RoomsRenderer* renderer = nullptr;

    // Controller UI
    ui::VContainer2D* right_hand_box = nullptr;
    ui::VContainer2D* left_hand_box = nullptr;

    // Main panel UI
    ui::HContainer2D* main_panel = nullptr;

    // Shortcuts
    virtual void generate_shortcuts() {}
    void update_shortcuts(const std::unordered_map<uint8_t, bool>& active_shortcuts);

    bool is_something_hovered();

public:

    BaseEditor() {};
    BaseEditor(const std::string& name) : name(name) {};

    virtual void initialize() {};
    virtual void clean();

    virtual void update(float delta_time);
    virtual void render();
    virtual void render_gui() {};

    virtual void on_enter(void* data) {};
    virtual void on_exit() {};

    void set_name(const std::string& new_name) { name = new_name; }
    const std::string& get_name() { return name; }
};
