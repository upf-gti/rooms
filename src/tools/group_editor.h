#pragma once

#include "tools/base_editor.h"

class Node;
class Node2D;
class Group3D;

namespace ui {
    class Inspector;
}

namespace shortcuts {
    enum : uint8_t {
        
    };
}

class GroupEditor : public BaseEditor {

    Group3D* current_group = nullptr;

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

    void select_node(Node* node, bool place = true);
    void deselect();
    void ungroup_node(Node* node);
    void process_node_hovered();

    /*
    *   UI
    */

    static uint64_t node_signal_uid;

    bool inspector_dirty = true;
    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;

    std::unordered_map<uint8_t, bool> shortcuts;

    void init_ui();
    void bind_events();

    void inspect_group(bool force = false);
    void inspect_node(Node* node, const std::string& texture_path = "");
    void update_panel_transform();

    bool rotation_started = false;
    bool scale_started = false;
    glm::quat last_left_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec3 last_left_hand_translation = {};
    float last_hand_distance = 0.0f;

    bool is_rotation_being_used();
    void update_node_transform();
    void update_hovered_node();

public:

    GroupEditor() {};
    GroupEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void on_enter(void* data) override;
    // void on_exit() override;

    void set_inspector_dirty() { inspector_dirty = true; };

    Group3D* get_current_group() { return current_group; }
};
