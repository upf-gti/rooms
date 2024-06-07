#pragma once

#include "base_editor.h"

#include "framework/nodes/ui.h"

class RoomsRenderer;
class Viewport3D;

class TutorialEditor : public BaseEditor {
protected:

    RoomsRenderer* renderer = nullptr;

    // Tutorial
    Node2D* xr_panel_2d = nullptr;
    Viewport3D* xr_panel_3d = nullptr;

    // Panels
    ui::XRPanel* welcome_panel = nullptr;
    ui::XRPanel* intro_panel = nullptr;
    ui::XRPanel* primitives_op_panel = nullptr;
    ui::XRPanel* materials_panel = nullptr;
    // ...

public:

    TutorialEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui()  override {};
};
