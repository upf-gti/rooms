#pragma once

#include "includes.h"

#include "framework/colors.h"
#include "framework/nodes/panel_2d.h"
#include "framework/nodes/node_2d.h"
#include "framework/nodes/container_2d.h"

#include "glm/vec2.hpp"

namespace ui {

    class XRPanel;

    using ContextMenuEvent = std::variant<FuncInt, bool*>;

    struct sContextMenuOption {
        std::string name;
        ContextMenuEvent event;
    };

    class ContextMenu : public Node2D {

        static uint32_t row_id;

        glm::vec2 panel_size = {};
        Color panel_color = { 0.01f, 0.01f, 0.01f, 0.95f };
        XRPanel* root = nullptr;
        VContainer2D* body = nullptr;

        uint32_t option_count = 0u;

        HContainer2D* create_row();

        void add_option(const sContextMenuOption& option);

    public:

        ContextMenu(const glm::vec2& position, const std::vector<sContextMenuOption>& new_options);
        ~ContextMenu();

        void update(float delta_time);
    };
}
