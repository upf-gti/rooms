#pragma once

#include "tools/base_editor.h"

#include "framework/math/transform.h"
#include "framework/colors.h"

#include <variant>

class Node;
class Node2D;
class Node3D;
class Group3D;
class Scene;
class Room;

namespace ui {
    class Inspector;
}

#define TIME_UNTIL_LONG_PRESS 0.15f

struct sActionData {

    using ActionDataParameter = std::variant<Transform, Node*, Node3D*, bool>;

    enum {
        ACTION_TRANSFORM,
        ACTION_DELETE,
        ACTION_GROUP,
        ACTION_UNGROUP,
        ACTION_CLONE
    };

    int type = -1;
    Node* ref_node = nullptr;
    ActionDataParameter param_1;
    ActionDataParameter param_2;
};

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;
    Room* current_room = nullptr;

    /*
    *   Intersections stuff
    */

    glm::vec3 ray_origin;
    glm::vec3 ray_direction;

    /*
    *   Input stuff
    */

    Node* selected_node = nullptr;
    Node* hovered_node = nullptr;

    bool select_action_pressed = false;

    /*
    *   Gizmo stuff
    */

    bool is_gizmo_usable();
    void update_gizmo(float delta_time);
    void render_gizmo();

    /*
    *   Node stuff
    */

    bool moving_node = false;

    void select_node(Node* node, bool place = true);
    void add_node(Node* node, Node* parent = nullptr, int idx = -1);
    void deselect();
    Node* clone_node(Node* node, bool copy = true, bool push_undo = true);
    void group_node(Node* node);
    void delete_node(Node* node, bool push_undo = true);
    void recover_node(Node* node, Node* parent = nullptr, bool push_redo = true);
    void ungroup_node(Node* node, bool push_undo = true, bool push_redo = true);
    void ungroup_all(Node* node);
    void create_light_node(uint8_t type);
    void process_node_hovered();

    /*
    *   Group stuff
    */

    bool grouping_node = false;
    Node* node_to_group = nullptr;
    Group3D* current_group = nullptr;

    void process_group(Node* node = nullptr, bool push_undo = true);
    void edit_group(Group3D* group);
    const Transform& get_group_global_transform(Node* node);

    /*
    *   UI
    */

    static uint64_t node_signal_uid;

    bool inspector_dirty = false;

    ui::Inspector* inspector = nullptr;

    std::unordered_map<uint8_t, bool> shortcuts;

    void init_ui();
    void bind_events();

    bool on_goback_inspector(ui::Inspector* scope);
    bool on_close_inspector(ui::Inspector* scope);

    void inspector_from_scene(bool force = false);
    void inspect_node(Node* node, uint32_t flags = NODE_STANDARD, const std::string& texture_path = "");
    void inspect_group(bool force = false);
    void inspect_light(bool force = false);

    bool rotation_started = false;
    bool scale_started = false;
    glm::quat last_right_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec3 last_right_hand_translation = {};
    float last_hand_distance = 0.0f;

    bool is_rotation_being_used();
    void update_node_transform(const float delta_time, const bool rotate_selected_node);
    void update_hovered_node();

    /*
    *   Room Player
    */

    void enter_room();

    /*
    *   Filesystem
    */

    bool exports_dirty = true;
    std::vector<std::string> exported_scenes;

    void get_export_files();
    void inspect_exports(bool force = false);

    /*
    *   Undo/Redo
    */

    bool action_in_progress = false;

    std::vector<sActionData> undo_list;
    std::vector<sActionData> redo_list;

    struct sDeletedNode {
        int index = -1;
        Node* ref = nullptr;
    };

    std::unordered_map<uintptr_t, sDeletedNode> deleted_nodes;

    bool scene_undo();
    bool scene_redo();

    void push_undo_action(const sActionData& step, bool clear_redo = true);
    void push_redo_action(const sActionData& step);

    enum eTriggerAction : uint8_t {
        NO_TRIGGER_ACTION = 0u,
        TRIGGER_TAPPED,
        TRIGGER_HOLDED
    };
    eTriggerAction prev_trigger_state = NO_TRIGGER_ACTION;
    Node* holded_node = nullptr;
    float time_pressed_storage = 0.0f;
    float hand_sculpt_distance = 0.0f;

    glm::vec3 prev_ray_origin = {};
    glm::vec3 prev_ray_dir = {};

    eTriggerAction get_trigger_action(const float delta_time);

public:

    static Color COLOR_HIGHLIGHT_NODE;
    static Color COLOR_HIGHLIGHT_GROUP;
    static Color COLOR_HIGHLIGHT_LIGHT;
    static Color COLOR_HIGHLIGHT_CHARACTER;

    SceneEditor() {};
    SceneEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    // void on_resize_window(uint32_t width, uint32_t height) override;
    void on_enter(void* data) override;

    void set_main_scene(Scene* new_scene) { main_scene = new_scene; };
    void set_inspector_dirty() { inspector_dirty = true; };

    Group3D* get_current_group() { return current_group; }
    Node* get_selected_node() { return selected_node; }

    static Color get_node_highlight_color(Node* node);
};
