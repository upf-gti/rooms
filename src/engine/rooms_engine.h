#pragma once

#include "engine/engine.h"

#include "framework/ui/gizmo_3d.h"
#include "framework/ui/cursor.h"

class Node;
class Node3D;
class MeshInstance3D;
class Environment3D;
class BaseEditor;
class SculptEditor;
class SceneEditor;
class TutorialEditor;
class SculptInstance;

enum EditorType : uint8_t {
    SCENE_EDITOR,
    SCULPT_EDITOR,
    SHAPE_EDITOR,
    TUTORIAL_EDITOR
};

#define _DESTROY_(x) if(x) { delete x; }

class RoomsEngine : public Engine
{
    static bool use_environment_map;
    static bool use_grid;

    // Editors

    bool tutorial_active = true;
    EditorType current_editor_type;

    BaseEditor* current_editor = nullptr;
    SculptEditor* sculpt_editor = nullptr;
    SceneEditor* scene_editor = nullptr;
    TutorialEditor* tutorial_editor = nullptr;

    void render_gui();

    // Engine meshes
    Environment3D* environment = nullptr;
    MeshInstance3D* grid = nullptr;
    MeshInstance3D* controller_mesh_left = nullptr;
    MeshInstance3D* controller_mesh_right = nullptr;
    MeshInstance3D* ray_pointer = nullptr;
    MeshInstance3D* sphere_pointer = nullptr;

    // make Cursor all static and remove this??
    ui::Cursor cursor;

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;

    void set_main_scene(const std::string& scene_path);
    void add_to_main_scene(const std::string& scene_path);

    bool show_tree_recursive(Node* entity);

    static void render_controllers();

    static void switch_editor(uint8_t editor);
    static void toggle_use_grid();
    static void toggle_use_environment_map();

    void set_current_sculpt(SculptInstance* sculpt_instance);

    void toggle_tutorial();

    inline BaseEditor* get_current_editor() const {
        return current_editor;
    }

    inline EditorType get_current_editor_type() const {
        return current_editor_type;
    }

    SculptEditor* get_sculpt_editor() const {
        return sculpt_editor;
    }
};
