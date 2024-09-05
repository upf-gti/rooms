#pragma once

#include "base_editor.h"

#include "framework/animation/track.h"
#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Keyframe;

namespace ui {
    class Inspector;
}

struct sAnimationState {
    struct sPropertyState {
        TrackType value;
        int track_id = -1;
    };

    std::unordered_map<std::string, sPropertyState> properties;
};

class AnimationEditor : public BaseEditor {

    Gizmo2D gizmo;

    Node3D* current_node = nullptr;
    Animation* current_animation = nullptr;
    Track* current_track = nullptr;
    Keyframe* current_keyframe = nullptr;
    uint32_t current_keyframe_idx = 0u;
    sAnimationState current_animation_properties;

    float current_time = 0.0f;
    bool keyframe_dirty = false;
    bool adding_keyframe = false;

    void add_keyframe();
    void process_keyframe();

    void store_animation_state(sAnimationState& state);

    /*
        UI
    */

    static uint64_t keyframe_signal_uid;
    static uint64_t node_signal_uid;

    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

    void init_ui();
    void bind_events();

    void inspect_keyframes_list(bool force = false);
    void inspect_keyframe();
    void inspect_keyframe_properties();
    void inspect_node(Node* node);

public:

    AnimationEditor() {};
    AnimationEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void on_enter(void* data);
    // void on_exit();
};
