#pragma once

#include "base_editor.h"

#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Track;

namespace ui {
    class Inspector;
}

struct sAnimationState {
    struct sPropertyState {
        std::variant<int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t, float, glm::vec2, glm::vec3, glm::vec4, glm::uvec2, glm::uvec3, glm::uvec4, glm::ivec2, glm::ivec3, glm::ivec4, glm::quat> value;
        int track_id = -1;
    };

    std::unordered_map<std::string, sPropertyState> properties;
};

class AnimationEditor : public BaseEditor {

    Gizmo2D gizmo;

    Animation* animation = nullptr;

    Track* current_track = nullptr;

    sAnimationState current_animation_properties;

    float current_time = 0.0f;

    /*
        UI
    */

    static uint64_t node_signal_uid;

    bool inspector_dirty = true;
    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

    void init_ui();
    void bind_events();

    void process_keyframe();

public:

    AnimationEditor() {};
    AnimationEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
