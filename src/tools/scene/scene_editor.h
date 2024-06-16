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
    NODE_STANDARD = NODE_NAME | NODE_VISIBILITY,
    NODE_LIGHT = NODE_NAME | NODE_VISIBILITY,
    NODE_SCULPT = NODE_NAME | NODE_VISIBILITY | NODE_EDIT
};

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;
    Node* selected_node = nullptr;

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
    void clone_node();

    void create_light_node(uint8_t type);

    /*
    *   UI
    */

    void init_ui();
    void bind_events();

    void inspector_from_scene();
    void inspect_node(Node* node, uint32_t flags = NODE_STANDARD, const std::string& texture_path = "");
    void inspect_light();

    static uint64_t node_signal_uid;

    bool inspector_dirty = true;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

public:

    SceneEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
