#pragma once

#include <string>

class RoomsRenderer;
class Node2D;
class Viewport3D;

namespace ui {
    class HContainer2D;
    class VContainer2D;
}

enum : uint32_t {
    LAYOUT_SCENE = 1 << 0,
    LAYOUT_MOVE_NODE = 1 << 1,
    LAYOUT_HOVER_NODE = 1 << 2,
    LAYOUT_SCULPT_NODE = 1 << 3,

    LAYOUT_SCULPT = 1 << 8,
    LAYOUT_PAINT = 1 << 9,
    LAYOUT_SPLINES = 1 << 10,

    LAYOUT_ANIMATION = 1 << 13,
    LAYOUT_KEYFRAME = 1 << 14,

    LAYOUT_SHIFT = 1 << 16,

    LAYOUT_SCENE_HOVER = LAYOUT_SCENE | LAYOUT_HOVER_NODE,

    //  to delete from here..

    LAYOUT_SCULPT_PAINT = LAYOUT_SCULPT | LAYOUT_PAINT,
    LAYOUT_ANIMATION_KEYFRAME = LAYOUT_ANIMATION | LAYOUT_KEYFRAME,
    LAYOUT_ANY = LAYOUT_SCENE | LAYOUT_SCULPT | LAYOUT_PAINT | LAYOUT_SPLINES | LAYOUT_ANIMATION,
    LAYOUT_SCENE_SHIFT = LAYOUT_SCENE | LAYOUT_SHIFT,
    LAYOUT_SCULPT_SHIFT = LAYOUT_SCULPT | LAYOUT_SHIFT,
    LAYOUT_PAINT_SHIFT = LAYOUT_PAINT | LAYOUT_SHIFT,
    LAYOUT_SCULPT_PAINT_SHIFT = LAYOUT_SCULPT_PAINT | LAYOUT_SHIFT,
    LAYOUT_SPLINES_SHIFT = LAYOUT_SPLINES | LAYOUT_SHIFT,
    LAYOUT_ANIMATION_SHIFT = LAYOUT_ANIMATION | LAYOUT_SHIFT,
    LAYOUT_ANY_SHIFT = LAYOUT_SCULPT_SHIFT | LAYOUT_PAINT_SHIFT | LAYOUT_SCULPT_PAINT_SHIFT | LAYOUT_ANIMATION_SHIFT,
    LAYOUT_ALL = 0xFFFFFFFF
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

    void update_controller_flags(uint32_t current_left_layout, uint32_t current_right_layout);

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
