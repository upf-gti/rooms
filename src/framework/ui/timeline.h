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
    class Timeline;
    class Keyframe;

    using TimelineFunc = std::function<bool(Timeline*)>;

    struct TimelineDesc {
        std::string name = "";
        std::string title = "";
        float title_height = 48.0f;
        float padding = 18.0f;
        glm::vec2 size = { 800.f, 190.f };
        glm::vec2 position = { 0.0f, 0.0f };
        TimelineFunc close_fn = nullptr;
        TimelineFunc edit_keyframe_fn = nullptr;
        TimelineFunc duplicate_keyframe_fn = nullptr;
        TimelineFunc delete_keyframe_fn = nullptr;
    };

    struct TimelineKeyframe {
        float time = 0.0f;
        Keyframe* keyframe = nullptr;
        uint32_t index = 0u;
        bool hovered = false;
        bool selected = false;
    };

    class Timeline : public Panel2D {

        /*
        *   Inner timeline
        */

        float padding = 0.0f;
        float zoom = 1.0f;
        float current_time = 0.0f;
        bool time_dirty = false;

        float x_to_time(float x);
        float time_to_x(float time);

        /*
        *   UI stuff
        */

        bool placed = false;
        bool grabbing = false;

        glm::vec2 panel_size = {};
        Color panel_color = { 0.01f, 0.01f, 0.01f, 0.95f };
        glm::vec2 last_scroll_position = {};
        glm::vec3 last_grab_position = {};

        XRPanel* root = nullptr;
        Container2D* body = nullptr;
        Panel2D* playhead = nullptr;
        Text2D* title = nullptr;
        Text2D* time_text = nullptr;
        TimelineFunc on_close = nullptr;
        TimelineFunc on_edit_keyframe = nullptr;
        TimelineFunc on_duplicate_keyframe = nullptr;
        TimelineFunc on_delete_keyframe = nullptr;

        /*
        *   Keyframes
        */

        TimelineKeyframe* selected_key = nullptr;

        std::vector<TimelineKeyframe> keyframes;

        Surface* quad_surface = nullptr;

        MeshInstance* frame_mesh = nullptr;
        MeshInstance* frame_mesh_selected = nullptr;
        MeshInstance* frame_mesh_hovered = nullptr;

        MeshInstance* generate_keyframe_mesh(const Color& color);
        void select_keyframe(TimelineKeyframe* key);

    public:

        Timeline() {};
        Timeline(const TimelineDesc& desc);
        ~Timeline();

        void render() override;
        void update(float delta_time) override;

        sInputData get_input_data(bool ignore_focus = false) override;

        bool is_time_dirty() { return time_dirty; }
        float get_current_time() { return current_time; }
        TimelineKeyframe* get_selected_key() { return selected_key; }

        void add_keyframe(float time, Keyframe* keyframe, uint32_t index);
        void clear();
        void set_title(const std::string& new_title);
        void set_current_time(float time) { current_time = time; }
    };
}
