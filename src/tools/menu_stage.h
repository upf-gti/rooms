#pragma once

#include "stage.h"

class RoomsRenderer;

namespace ui {
    class XRPanel;
};

enum eMenuSection : uint8_t {
    MENU_SECTION_MAIN,
    MENU_SECTION_DISCOVER,
    MENU_SECTION_COUNT
};

class MenuStage : public Stage {
protected:

    RoomsRenderer* renderer = nullptr;

    bool placed         = false;
    bool grabbing       = false;

    glm::vec3 last_grab_position;

    // Tutorial
    Node2D* panel = nullptr;
    ui::XRPanel* current_panel = nullptr;
    uint32_t current_panel_idx = 0u;

    // Panels
    ui::XRPanel* panels[MENU_SECTION_COUNT];
    void generate_section(const std::string& name, const std::string& path, uint8_t section_idx);

public:

    MenuStage() {};
    MenuStage(const std::string& name) : Stage(name) {};

    void initialize() override;

    void update(float delta_time) override;
    void render() override;
};
