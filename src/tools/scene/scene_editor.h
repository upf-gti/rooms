#pragma once

#include "tools/base_editor.h"

#include "framework/ui/gizmo_3d.h"

class MeshInstance3D;
class Scene;

class SceneEditor : public BaseEditor {

    Scene* main_scene = nullptr;

    static Gizmo3D gizmo;

public:

    SceneEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
