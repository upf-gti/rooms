#pragma once

#include "framework/nodes/ui.h"

class RoomsRenderer;
class Viewport3D;

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

public:

    BaseEditor() {};

    virtual void initialize() {};
    virtual void clean() {};

    virtual void update(float delta_time) {};
    virtual void render() {};
    virtual void render_gui() {};
};
