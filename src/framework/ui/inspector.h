#pragma once

#include "includes.h"

#include "framework/colors.h"
#include "framework/nodes/panel_2d.h"
#include "framework/nodes/node_2d.h"
#include "framework/nodes/container_2d.h"

#include "glm/vec2.hpp"

namespace ui {

    class Text2D;
    class XRPanel;
    class TextureButton2D;
    class Inspector;

    using InspectorFunc = std::function<bool(Inspector*)>;

    struct InspectorDesc {
        std::string name = "";
        std::string title = "";
        float title_height = 48.0f;
        float padding = 18.0f;
        glm::vec2 size = { 420.f, 600.f };
        glm::vec2 position = { 0.0f, 0.0f };
        InspectorFunc close_fn = nullptr;
        InspectorFunc back_fn = nullptr;
    };

    enum eInspectorResetFlags : uint8_t {
        INSPECTOR_FLAG_FORCE_3D_POSITION = 1 << 0,
        INSPECTOR_FLAG_BACK_BUTTON = 1 << 1,
        INSPECTOR_FLAG_CLOSE_BUTTON = 1 << 2,
    };

    class Inspector : public Node2D {

        static uint32_t row_id;

        glm::vec2 panel_size = {};
        Color panel_color = { 0.01f, 0.01f, 0.01f, 0.95f };

        bool placed = false;
        bool grabbing = false;

        float body_height = 0.0f;
        float padding = 0.0f;
        float scroll_top = 0.0f;

        uint32_t indentation_level = 0u;

        XRPanel* root = nullptr;
        VContainer2D* body = nullptr;
        Text2D* title = nullptr;
        HContainer2D* current_row = nullptr;
        TextureButton2D* back_button = nullptr;
        TextureButton2D* close_button = nullptr;

        HContainer2D* create_row();

        InspectorFunc on_close = nullptr;
        InspectorFunc on_back = nullptr;

        glm::vec2 last_scroll_position = {};
        glm::vec3 last_grab_position = {};

        std::map<std::string, Node2D*> items;
        std::map<std::string, std::variant<glm::fvec2, glm::fvec3, glm::fvec4, glm::ivec2, glm::ivec3, glm::ivec4>> inner_data;

    public:

        Inspector() {};
        Inspector(const InspectorDesc& desc);
        ~Inspector();

        void update(float delta_time);
        void clear(uint8_t reset_flags = INSPECTOR_FLAG_CLOSE_BUTTON, const std::string& new_title = "");
        void clear_scroll();

        void set_title(const std::string& new_title);

        void label(const std::string& name, const std::string& text, uint32_t flags = 0, const Color& c = colors::BLACK);
        void icon(const std::string& texture_path);
        void button(const std::string& name, const std::string& texture_path, uint32_t flags = 0, const std::string& label = "");

        void color_picker(const std::string& name, const Color& c, Color* result = nullptr);

        void fslider(const std::string& name, float value, float* result = nullptr, float min = 0.0f, float max = 1.0f, int precision = 1);
        void islider(const std::string& name, int value, int* result = nullptr, int min = -10, int max = 10);

        void create_vector_component(ui::HContainer2D* container, const std::string& name, const std::string& value, char component, uint32_t flags = 0);

        template <typename T>
        void vector2(const std::string& name, glm::vec<2, T> value, T min, T max, glm::vec<2, T>* result = nullptr)
        {
            ui::HContainer2D* flex_container = current_row;

            if (!flex_container) {
                flex_container = create_row();
            }

            create_vector_component(flex_container, name, std::to_string(value.x), 'x');
            create_vector_component(flex_container, name, std::to_string(value.y), 'y');

            // Store value in inner data
            inner_data[name] = value;

            if (result != nullptr) {
                Node::bind(name + "_x@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<2, T> value = std::get<glm::vec<2, T>>(inner_data[name]);
                    (*result).x += dt;
                    std::string new_value = std::to_string((*result).x);
                    Text2D* w = static_cast<Text2D*>(items[name + "_x"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_y@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<2, T> value = std::get<glm::vec<2, T>>(inner_data[name]);
                    (*result).y += dt;
                    std::string new_value = std::to_string((*result).y);
                    Text2D* w = static_cast<Text2D*>(items[name + "_y"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });
            }
        }

        template <typename T>
        void vector3(const std::string& name, glm::vec<3, T> value, T min, T max, glm::vec<3, T>* result = nullptr)
        {
            ui::HContainer2D* flex_container = current_row;

            if (!flex_container) {
                flex_container = create_row();
            }

            create_vector_component(flex_container, name, std::to_string(value.x), 'x');
            create_vector_component(flex_container, name, std::to_string(value.y), 'y');
            create_vector_component(flex_container, name, std::to_string(value.z), 'z');

            // Store value in inner data
            inner_data[name] = value;

            if (result != nullptr) {
                Node::bind(name + "_x@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<3, T> value = std::get<glm::vec<3, T>>(inner_data[name]);
                    (*result).x += dt;
                    std::string new_value = std::to_string((*result).x);
                    Text2D* w = static_cast<Text2D*>(items[name + "_x"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_y@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<3, T> value = std::get<glm::vec<3, T>>(inner_data[name]);
                    (*result).y += dt;
                    std::string new_value = std::to_string((*result).y);
                    Text2D* w = static_cast<Text2D*>(items[name + "_y"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_z@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<3, T> value = std::get<glm::vec<3, T>>(inner_data[name]);
                    (*result).z += dt;
                    std::string new_value = std::to_string((*result).z);
                    Text2D* w = static_cast<Text2D*>(items[name + "_z"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });
            }
        }

        template <typename T>
        void vector4(const std::string& name, glm::vec<4, T> value, T min, T max, glm::vec<4, T>* result = nullptr)
        {
            ui::HContainer2D* flex_container = current_row;

            if (!flex_container) {
                flex_container = create_row();
            }

            create_vector_component(flex_container, name, std::to_string(value.x), 'x');
            create_vector_component(flex_container, name, std::to_string(value.y), 'y');
            create_vector_component(flex_container, name, std::to_string(value.z), 'z');
            create_vector_component(flex_container, name, std::to_string(value.w), 'w');

            // Store value in inner data
            inner_data[name] = value;

            if (result != nullptr) {
                Node::bind(name + "_x@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<4, T> value = std::get<glm::vec<4, T>>(inner_data[name]);
                    (*result).x += dt;
                    std::string new_value = std::to_string((*result).x);
                    Text2D* w = static_cast<Text2D*>(items[name + "_x"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_y@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<4, T> value = std::get<glm::vec<4, T>>(inner_data[name]);
                    (*result).y += dt;
                    std::string new_value = std::to_string((*result).y);
                    Text2D* w = static_cast<Text2D*>(items[name + "_y"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_z@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<4, T> value = std::get<glm::vec<4, T>>(inner_data[name]);
                    (*result).z += dt;
                    std::string new_value = std::to_string((*result).z);
                    Text2D* w = static_cast<Text2D*>(items[name + "_z"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });

                Node::bind(name + "_w@stick_moved", (std::function<void(const std::string&, T)>)[&, result = result](const std::string& signal, float dt) {
                    std::string name = signal.substr(0, signal.find('@') - 2);
                    glm::vec<4, T> value = std::get<glm::vec<4, T>>(inner_data[name]);
                    (*result).w += dt;
                    std::string new_value = std::to_string((*result).w);
                    Text2D* w = static_cast<Text2D*>(items[name + "_w"]);
                    w->set_text(new_value.substr(0, new_value.find('.') + 3));
                    Node::emit_signal(name + "@changed", (void*)nullptr);
                });
            }
        }

        void same_line();
        void end_line();
        void set_indentation(uint32_t level);

        Node2D* get(const std::string& name);
    };
}
