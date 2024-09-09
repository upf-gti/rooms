#pragma once

#include "base_editor.h"

#include "framework/animation/track.h"
#include "framework/ui/gizmo_2d.h"
#include "framework/ui/gizmo_3d.h"

class RoomsRenderer;
class Node;
class Node3D;
class Animation;
class Keyframe;
class Viewport3D;

namespace ui {
    class Inspector;
};

struct sPropertyState {
    TrackType value;
    int track_id = -1;
    Keyframe* keyframe = nullptr;
};

struct sAnimationState {
    float time = 0.0f;
    std::unordered_map<std::string, sPropertyState> properties;
};

class AnimationEditor : public BaseEditor {

    Gizmo2D gizmo_2d;
    Gizmo3D gizmo_3d;

    Node3D* current_node = nullptr;
    Animation* current_animation = nullptr;
    Track* current_track = nullptr;

    sAnimationState* current_animation_state = nullptr;

    std::vector<sAnimationState> animation_states;

    float current_time = 0.0f;
    bool show_keyframe_dirty = false;
    bool editing_keyframe = false;

    bool  is_editing = false;

    void create_keyframe();
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
