#pragma once

#include "base_editor.h"

#include "framework/nodes/ui.h"

class RoomsRenderer;
class Viewport3D;

enum : uint8_t {
    TUTORIAL_NONE,
    TUTORIAL_WELCOME,
    TUTORIAL_STAMP_SMEAR,
    TUTORIAL_PRIMITIVES_OPERATIONS,
    TUTORIAL_CURVES,
    TUTORIAL_GUIDES,
    TUTORIAL_MATERIAL,
    TUTORIAL_PAINT,
    TUTORIAL_UNDO_REDO,
    TUTORIAL_PANEL_COUNT,
};

class TutorialEditor : public BaseEditor {
protected:

    RoomsRenderer* renderer = nullptr;

    // Tutorial
    Node2D* xr_panel_2d = nullptr;
    Viewport3D* xr_panel_3d = nullptr;

    // Panels
    ui::XRPanel* panels[TUTORIAL_PANEL_COUNT];

    ui::XRPanel* generate_panel(const std::string& name, const std::string& path, uint8_t prev, uint8_t next);

public:

    TutorialEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui()  override {};
};
