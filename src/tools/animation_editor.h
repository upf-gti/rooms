#pragma once

#include "base_editor.h"

#include "framework/animation/track.h"

class RoomsRenderer;
class Node;
class Node3D;
class MeshInstance3D;
class Joint3D;
class Animation;
class Surface;

namespace ui {
    class Inspector;
};

namespace shortcuts {
    enum : uint8_t {
        OPEN_KEYFRAME_LIST,
        CREATE_KEYFRAME,
        SUBMIT_KEYFRAME,
        PLAY_ANIMATION
    };
}

struct sPropertyState {
    TrackType value;
    int track_id = -1;
    Keyframe* keyframe = nullptr;
};

struct sAnimationState {
    float time = 0.0f;
    std::unordered_map<std::string, sPropertyState> properties;
};

struct sAnimationData {
    float current_time = 0.0f;
    Animation* animation = nullptr;
    std::vector<sAnimationState> states;
};

class AnimationEditor : public BaseEditor {

    Node3D* current_node = nullptr;
    Joint3D* current_joint = nullptr;
    Animation* current_animation = nullptr;
    Track* current_track = nullptr;

    sAnimationState* current_animation_state = nullptr;

    std::unordered_map<uint32_t, sAnimationData> animations_data;

    Surface* animation_trajectory_mesh = nullptr;
    MeshInstance3D* animation_trajectory_instance = nullptr;
    MeshInstance3D* keyframe_markers_render_instance = nullptr;

    void render_gizmo();
    void update_gizmo(float delta_time);

    void update_animation_trajectory();

    uint32_t get_animation_idx();

    /*
    *   Keyframes
    */

    float current_time = 0.0f;
    bool show_keyframe_dirty = false;
    bool keyframe_list_dirty = false;
    bool keyframe_dirty = false;
    bool editing_keyframe = false;

    void create_keyframe();
    void process_keyframe();
    void edit_keyframe(uint32_t index);
    void duplicate_keyframe(uint32_t index);

    void set_animation_state(uint32_t index);
    void store_animation_state(sAnimationState& state);
    sAnimationState* get_animation_state(uint32_t index);

    /*
        Animation Player
    */

    void set_loop_type(uint8_t type);

    void play_animation();
    void pause_animation();
    void stop_animation();

    /*
        UI
    */

    static uint64_t keyframe_signal_uid;
    static uint64_t node_signal_uid;

    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;

    void init_ui();
    void bind_events();
    void update_panel_transform();
    void generate_shortcuts() override;

    void inspect_keyframes_list(bool force = false);
    void inspect_keyframe();
    void inspect_keyframe_properties();
    void inspect_node(Node* node);

    bool on_close_inspector(ui::Inspector* scope = nullptr);

public:

    AnimationEditor() {};
    AnimationEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void on_enter(void* data) override;
    void on_exit() override;
};
