#pragma once

#include "tools/base_editor.h"

#include "framework/ui/gizmo_2d.h"
#include "framework/ui/gizmo_3d.h"

class Node;
class Node2D;
class MeshInstance3D;
class Scene;

namespace ui {
    class Inspector;
}

enum InspectNodeFlags {
    NODE_NAME = 1 << 0,
    NODE_ICON = 1 << 1,
    NODE_VISIBILITY = 1 << 2,
    NODE_EDIT = 1 << 3,
    NODE_ANIMATE = 1 << 4,
    NODE_STANDARD = NODE_NAME | NODE_VISIBILITY | NODE_ANIMATE,
    NODE_LIGHT = NODE_STANDARD,
    NODE_SCULPT = NODE_STANDARD | NODE_EDIT
};

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;

    Node* selected_node = nullptr;
    Node* hovered_node = nullptr;

    /*
    *   Gizmo stuff
    */

    Gizmo2D gizmo_2d;
    Gizmo3D gizmo_3d;

    bool is_gizmo_usable();
    void update_gizmo(float delta_time);
    void render_gizmo();

    void set_gizmo_translation();
    void set_gizmo_rotation();
    void set_gizmo_scale();

    /*
    *   Node actions
    */

    bool moving_node = false;

    void select_node(Node* node, bool place = true);
    void clone_node(Node* node, bool copy = true);

    void create_light_node(uint8_t type);

    /*
    *   UI
    */

    static uint64_t node_signal_uid;

    bool inspector_dirty = true;
    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

    void init_ui();
    void bind_events();

    void inspector_from_scene(bool force = false);
    void inspect_node(Node* node, uint32_t flags = NODE_STANDARD, const std::string& texture_path = "");
    void inspect_light();
    void update_panel_transform();

    bool rotation_started = false;
    glm::quat last_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec3 last_hand_translation = {};

    bool is_rotation_being_used();
    void update_node_rotation();


    /*
    *   Filesystem
    */

    bool exports_dirty = true;
    std::vector<std::string> exported_scenes;

    void get_export_files();
    void inspect_exports(bool force = false);

public:

    SceneEditor() {};
    SceneEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void set_main_scene(Scene* new_scene) { main_scene = new_scene; };
    void set_inspector_dirty() { inspector_dirty = true; };

    void update_hovered_node();
};
