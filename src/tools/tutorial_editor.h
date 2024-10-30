#pragma once

#include "base_editor.h"

class RoomsRenderer;

namespace ui {
    class XRPanel;
};

enum : uint8_t {
    TUTORIAL_NONE,
    TUTORIAL_WELCOME,
    TUTORIAL_SCENE_1,
    TUTORIAL_SCENE_2,
    TUTORIAL_SCENE_3,
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

    bool placed = false;
    bool grabbing = false;

    // Tutorial
    Node2D* panel = nullptr;
    ui::XRPanel* current_panel = nullptr;

    // Panels
    ui::XRPanel* panels[TUTORIAL_PANEL_COUNT];
    ui::XRPanel* generate_panel(const std::string& name, const std::string& path, uint8_t prev, uint8_t next);

public:

    TutorialEditor() {};
    TutorialEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui()  override {};

    void generate_shortcuts() override {};
};
