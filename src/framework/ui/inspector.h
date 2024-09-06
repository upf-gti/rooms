#pragma once

#include "includes.h"

#include "framework/colors.h"
#include "framework/nodes/node_2d.h"

#include "glm/vec2.hpp"

namespace ui {

    class XRPanel;
    class VContainer2D;
    class HContainer2D;

    struct InspectorDesc {
        std::string name = "";
        std::string title = "";
        float title_height = 48.0f;
        float padding = 18.0f;
        glm::vec2 size = { 420.f, 600.f };
        glm::vec2 position = { 0.0f, 0.0f };
    };

    class Inspector : public Node2D {

        static uint32_t row_id;

        glm::vec2 panel_size = {};
        Color panel_color = { 0.01f, 0.01f, 0.01f, 0.95f };

        float body_height = 0.0f;
        float padding = 0.0f;
        float scroll_top = 0.0f;

        XRPanel* root = nullptr;
        VContainer2D* body = nullptr;
        HContainer2D* current_row = nullptr;

        HContainer2D* create_row();

        glm::vec2 last_grab_position = {};

        std::map<std::string, Node2D*> items;

    public:

        Inspector() {};
        Inspector(const InspectorDesc& desc);

        void update(float delta_time);
        /*void render(); */
        void clear();

        void label(const std::string& name, const std::string& label, uint32_t flags = 0);
        void icon(const std::string& texture_path);
        void button(const std::string& name, const std::string& texture_path, uint32_t flags = 0);

        void color_picker(const std::string& name, const Color& c, Color* result = nullptr);

        void fslider(const std::string& name, float value, float* result = nullptr, float min = 0.0f, float max = 1.0f, int precision = 1);
        void islider(const std::string& name, int value, int* result = nullptr, int min = -10, int max = 10);

        void float2(const std::string& name, glm::vec2 value, glm::vec2* result = nullptr, float min = 0.0f, float max = 1.0f);
        void float3(const std::string& name, glm::vec3 value, glm::vec3* result = nullptr, float min = 0.0f, float max = 1.0f);

        void same_line();
        void end_line();

        Node2D* get(const std::string& name);
    };
}
