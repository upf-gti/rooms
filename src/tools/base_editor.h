#pragma once

#include <string>
#include <unordered_map>

class RoomsRenderer;
class Node2D;
class Viewport3D;

namespace ui {
    class HContainer2D;
    class VContainer2D;
}

namespace shortcuts {
    // Left hand
    const std::string X_BUTTON_PATH = "data/textures/buttons/x.png";
    const std::string Y_BUTTON_PATH = "data/textures/buttons/y.png";

    // Right hand
    const std::string A_BUTTON_PATH = "data/textures/buttons/a.png";
    const std::string B_BUTTON_PATH = "data/textures/buttons/b.png";
}

class BaseEditor {

    std::string name;

protected:

    bool is_shift_left_pressed = false;
    bool is_shift_right_pressed = false;

    RoomsRenderer* renderer = nullptr;

    // Controller UI
    ui::VContainer2D* right_hand_box = nullptr;
    ui::VContainer2D* left_hand_box = nullptr;
    Viewport3D* right_hand_ui_3D = nullptr;
    Viewport3D* left_hand_ui_3D = nullptr;

    // Main panel UI
    ui::HContainer2D* main_panel_2d = nullptr;
    Viewport3D* main_panel_3d = nullptr;

    // Tutorial
    Node2D* xr_panel_2d = nullptr;
    Viewport3D* xr_panel_3d = nullptr;

    // Shortcuts
    virtual void generate_shortcuts() = 0;
    void update_shortcuts(const std::unordered_map<uint8_t, bool>& active_shortcuts);

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
