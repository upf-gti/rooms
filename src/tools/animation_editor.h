#pragma once

#include "base_editor.h"

#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Track;
class Keyframe;

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
    Keyframe* current_keyframe = nullptr;
    uint32_t current_keyframe_idx = 0u;
    sAnimationState current_animation_properties;

    float current_time = 0.0f;
    bool keyframe_dirty = false;

    void process_keyframe();

    /*
        UI
    */

    static uint64_t keyframe_signal_uid;

    bool inspector_dirty = true;
    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

    void init_ui();
    void bind_events();

    void inspect_keyframes_list(bool force = false);
    void inspect_keyframe();

public:

    AnimationEditor() {};
    AnimationEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
