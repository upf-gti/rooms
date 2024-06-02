#pragma once

#include "tools/base_editor.h"

#include "framework/ui/gizmo_2d.h"
#include "framework/ui/gizmo_3d.h"

class Node;
class MeshInstance3D;
class Scene;

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;
    Node* selected_node = nullptr;

    /*
    *   Gizmo stuff
    */

    Gizmo3D gizmo_3d;
    Gizmo2D gizmo_2d;

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

    void add_node(Node* node);
    void clone_node();

    Node* create_light_node(uint8_t type);

    /*
    *   UI
    */

    void init_ui();
    void bind_events();

public:

    SceneEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
