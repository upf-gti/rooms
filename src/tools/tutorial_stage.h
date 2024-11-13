#pragma once

#include "stage.h"

class RoomsRenderer;

namespace ui {
    class XRPanel;
};

class TutorialStage : public Stage {
protected:

    RoomsRenderer* renderer = nullptr;

    bool active         = true;
    bool placed         = false;
    bool grabbing       = false;

    glm::vec3 last_grab_position;

    // Tutorial
    Node2D* panel = nullptr;
    ui::XRPanel* current_panel = nullptr;
    uint32_t current_panel_idx = 0u;
    void next_panel();

    // Panels
    ui::XRPanel* panels[TUTORIAL_PANEL_COUNT];
    ui::XRPanel* generate_panel(const std::string& name, const std::string& path, uint8_t prev, uint8_t next);

public:

    TutorialStage() {};
    TutorialStage(const std::string& name) : Stage(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui()  override {};

    void generate_shortcuts() override {};

    void end();
};
