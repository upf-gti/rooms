#pragma once

#include "base_editor.h"

#include "framework/animation/animation.h"
#include "framework/animation/skeleton.h"

class RoomsRenderer;
class Node;
class Node3D;
class MeshInstance3D;
class SkeletonInstance3D;
class Joint3D;
class Surface;

namespace ui {
    class Inspector;
    class Timeline;
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
    int keyframe_idx = -1;
};

struct sAnimationState {
    float time = 0.0f;
    std::unordered_map<std::string, sPropertyState> properties;
};

struct sAnimationData {
    Animation* animation = nullptr;
    std::vector<sAnimationState> states;
};

using NodeAnimationsMap = std::unordered_map<std::string, sAnimationData>;

struct sIKChain {
    std::vector<Transform> transforms;
    std::vector<uint32_t> indices;

    void push(const Transform& t, uint32_t idx) {
        transforms.push_back(t);
        indices.push_back(idx);
    }
};

class AnimationEditor : public BaseEditor {

    /*
    *   Animation stuff
    */

    Animation* current_animation = nullptr;
    int last_animation_state_idx = -1;

    // this stores a list of animation data per node name
    std::unordered_map<std::string, NodeAnimationsMap> animation_storage;

    Animation* create_new_animation();
    void set_animation(const std::string& name);

    /*
    *   Nodes
    */

    Surface* animation_trajectory_mesh = nullptr;
    MeshInstance3D* animation_trajectory_instance = nullptr;
    MeshInstance3D* keyframe_markers_render_instance = nullptr;

    Node3D* current_node = nullptr;
    Joint3D* current_joint = nullptr;

    bool rotation_started = false;
    bool scale_started = false;
    glm::quat last_left_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec3 last_left_hand_translation = {};
    float last_hand_distance = 0.0f;

    Node3D* get_current_node();
    SkeletonInstance3D* find_skeleton(Node* node);

    void update_node_from_state(uint32_t index);
    void update_node_transform();
    void on_select_joint();
    void save_character_animation();

    /*
    *   Keyframes
    */

    float current_time = 0.0f;
    bool show_keyframe_dirty = false;
    bool keyframe_list_dirty = false;
    bool keyframe_dirty = false;
    bool editing_keyframe = false;

    void create_keyframe();
    bool insert_keyframe();
    void process_keyframe();
    bool edit_keyframe(uint32_t index);
    bool duplicate_keyframe(uint32_t index);
    void delete_keyframe(uint32_t index);

    void set_animation_state(uint32_t index);
    void store_animation_state(sAnimationState& state);
    std::vector<sAnimationState>& get_animation_states();
    sAnimationState* get_animation_state(uint32_t index);

    /*
    *   IK
    */

    bool ik_enabled = true;
    bool ik_active = false;

    std::unordered_map<uint8_t, sIKChain> ik_chains;

    void initialize_ik();
    void create_ik_chain(uint32_t chain_endpoint_idx, uint32_t depth);
    void set_active_chain(uint32_t chain_idx);
    void update_ik(float delta_time);

    /*
        Animation Player
    */

    void set_loop_type(uint8_t type);

    void play_animation();
    void pause_animation();
    void stop_animation();

    /*
        Timeline
    */

    bool timeline_dirty = true;
    void update_timeline();

    /*
        UI
    */

    static uint64_t keyframe_signal_uid;
    static uint64_t node_signal_uid;

    bool gizmo_active = false;
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

    bool on_move_timeline_keyframe(ui::Timeline* scope, uint32_t index);
    bool on_delete_timeline_keyframe(ui::Timeline* scope, uint32_t index);

    bool on_close_inspector(ui::Inspector* scope = nullptr);

    void render_gizmo();
    void update_gizmo(float delta_time);
    void update_animation_trajectory();

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
