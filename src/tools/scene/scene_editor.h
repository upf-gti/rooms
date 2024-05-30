#pragma once

#include "tools/base_editor.h"

#include "framework/ui/gizmo_2d.h"
#include "framework/ui/gizmo_3d.h"

class Node;
class MeshInstance3D;
class Scene;

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;

    static Gizmo3D gizmo_3d;

    static Gizmo2D gizmo_2d;

    Node* selected_node = nullptr;

    void init_ui();
    void bind_events();

    void clone_node();

    /*
    *   Gizmo stuff
    */

    bool is_gizmo_usable();
    void update_gizmo(float delta_time);
    void render_gizmo();

    void set_gizmo_translation();
    void set_gizmo_rotation();
    void set_gizmo_scale();

public:

    SceneEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
