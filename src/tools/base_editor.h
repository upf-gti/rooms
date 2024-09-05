#pragma once

#include "framework/nodes/ui.h"

class RoomsRenderer;
class Viewport3D;

enum : uint8_t {
    LAYOUT_SCENE = 1 << 0,
    LAYOUT_CLONE = 1 << 1,
    LAYOUT_SCULPT = 1 << 2,
    LAYOUT_PAINT = 1 << 3,
    LAYOUT_SPLINES = 1 << 4,
    LAYOUT_SHIFT = 1 << 5,
    LAYOUT_SCULPT_PAINT = LAYOUT_SCULPT | LAYOUT_PAINT,
    LAYOUT_ANY = LAYOUT_SCENE | LAYOUT_SCULPT | LAYOUT_PAINT | LAYOUT_SPLINES,
    LAYOUT_SCENE_SHIFT = LAYOUT_SCENE | LAYOUT_SHIFT,
    LAYOUT_CLONE_SHIFT = LAYOUT_CLONE | LAYOUT_SHIFT,
    LAYOUT_SCULPT_SHIFT = LAYOUT_SCULPT | LAYOUT_SHIFT,
    LAYOUT_PAINT_SHIFT = LAYOUT_PAINT | LAYOUT_SHIFT,
    LAYOUT_SCULPT_PAINT_SHIFT = LAYOUT_SCULPT_PAINT | LAYOUT_SHIFT,
    LAYOUT_SPLINES_SHIFT = LAYOUT_SPLINES | LAYOUT_SHIFT,
    LAYOUT_ANY_SHIFT = LAYOUT_SCULPT_SHIFT | LAYOUT_PAINT_SHIFT | LAYOUT_SCULPT_PAINT_SHIFT,
    LAYOUT_ALL = 0xFF
};

class BaseEditor {

    std::string name;

protected:

    bool is_shift_left_pressed = false;
    bool is_shift_right_pressed = false;

    RoomsRenderer* renderer = nullptr;

    // Controller UI
    ui::VContainer2D* right_hand_container = nullptr;
    ui::VContainer2D* left_hand_container = nullptr;
    Viewport3D* right_hand_ui_3D = nullptr;
    Viewport3D* left_hand_ui_3D = nullptr;

    // Main panel UI
    ui::HContainer2D* main_panel_2d = nullptr;
    Viewport3D* main_panel_3d = nullptr;

    // Tutorial
    Node2D* xr_panel_2d = nullptr;
    Viewport3D* xr_panel_3d = nullptr;

    void update_controller_flags(uint8_t current_layout);
    bool should_render_label(uint8_t mask, uint8_t state);

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
