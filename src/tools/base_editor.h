#pragma once

#include "framework/nodes/ui.h"

class RoomsRenderer;
class Viewport3D;

enum EditorLayout : uint8_t {
    LAYOUT_NONE = 1 << 0,
    LAYOUT_SCULPT = 1 << 1,
    LAYOUT_PAINT = 1 << 2,
    LAYOUT_SHIFT_R = 1 << 3,
    LAYOUT_SHIFT_L = 1 << 4,
    LAYOUT_NO_SHIFT_R = 1 << 5,
    LAYOUT_NO_SHIFT_L = 1 << 6,
    LAYOUT_SCULPT_SHIFT_R = LAYOUT_SCULPT | LAYOUT_SHIFT_R,
    LAYOUT_SCULPT_SHIFT_L = LAYOUT_SCULPT | LAYOUT_SHIFT_L,
    LAYOUT_SCULPT_NO_SHIFT_R = LAYOUT_SCULPT | LAYOUT_NO_SHIFT_R,
    LAYOUT_SCULPT_NO_SHIFT_L = LAYOUT_SCULPT | LAYOUT_NO_SHIFT_L,
    LAYOUT_PAINT_SHIFT_R = LAYOUT_PAINT | LAYOUT_SHIFT_R,
    LAYOUT_PAINT_SHIFT_L = LAYOUT_PAINT | LAYOUT_SHIFT_L,
    LAYOUT_PAINT_NO_SHIFT_R = LAYOUT_PAINT | LAYOUT_NO_SHIFT_R,
    LAYOUT_PAINT_NO_SHIFT_L = LAYOUT_PAINT | LAYOUT_NO_SHIFT_L,
    LAYOUT_ANY_SHIFT_R = LAYOUT_SCULPT_SHIFT_R | LAYOUT_PAINT_SHIFT_R,
    LAYOUT_ANY_SHIFT_L = LAYOUT_SCULPT_SHIFT_L | LAYOUT_PAINT_SHIFT_L,
    LAYOUT_ANY_NO_SHIFT_R = LAYOUT_SCULPT_NO_SHIFT_R | LAYOUT_PAINT_NO_SHIFT_R,
    LAYOUT_ANY_NO_SHIFT_L = LAYOUT_SCULPT_NO_SHIFT_L | LAYOUT_PAINT_NO_SHIFT_L,
    LAYOUT_SCULPT_ALL = (LAYOUT_SCULPT_SHIFT_R | LAYOUT_SCULPT_SHIFT_L) | (LAYOUT_SCULPT_NO_SHIFT_R | LAYOUT_SCULPT_NO_SHIFT_L),
    LAYOUT_PAINT_ALL = (LAYOUT_PAINT_SHIFT_R | LAYOUT_PAINT_SHIFT_L) | (LAYOUT_PAINT_NO_SHIFT_R | LAYOUT_PAINT_NO_SHIFT_L),
    LAYOUT_ALL = LAYOUT_SCULPT_ALL | LAYOUT_PAINT_ALL
};

class BaseEditor {
protected:

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

public:

    BaseEditor() {};

    virtual void initialize() {};
    virtual void clean();

    virtual void update(float delta_time);
    virtual void render();
    virtual void render_gui() {};
};
