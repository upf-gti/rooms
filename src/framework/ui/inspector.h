#pragma once

#include "includes.h"

#include "framework/nodes/node_2d.h"
#include "framework/nodes/ui.h"

#include "glm/vec2.hpp"

namespace ui {

    struct InspectorDesc {
        std::string name = "";
        std::string title = "";
        float title_height = 48.0f;
        float padding = 12.0f;
        glm::vec2 size = { 300, 400.f };
        glm::vec2 position = { 0.0f, 0.0f };
    };

    class Inspector : public Node2D {

        static uint32_t row_id;

        glm::vec2 panel_size = {};
        Color panel_color = { 0.0f, 0.0f, 0.0f, 0.8f };
        float padding = 0.0f;
        float scroll_top = 0.0f;

        ui::XRPanel* root = nullptr;
        VContainer2D* body = nullptr;
        HContainer2D* current_row = nullptr;

        HContainer2D* create_row();

        glm::vec2 last_grab_position = {};

    public:

        Inspector() {};
        Inspector(const InspectorDesc& desc);

        void update(float delta_time);
        /*void render(); */

        void add_label(const std::string& label);
        void add_button(const std::string& name, const std::string& texture_path, uint32_t flags = 0);
        void add_slider(const std::string& name, float value, float min = 0.0f, float max = 1.0f, int precision = 1);

        void same_line();
        void end_line();
    };
}