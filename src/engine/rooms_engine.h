#pragma once

#include "engine/engine.h"

#include "framework/ui/gizmo_3d.h"
#include "framework/ui/cursor.h"
#include "framework/animation/skeleton.h"

class Node;
class Node3D;
class SculptNode;
class Environment3D;
class BaseEditor;

namespace ui {
    class ContextMenu;
}

enum EditorType : uint8_t {
    SCENE_EDITOR,
    SCULPT_EDITOR,
    ANIMATION_EDITOR,
    TUTORIAL_EDITOR,
    PLAYER_EDITOR,
    EDITOR_COUNT
};

class RoomsEngine : public Engine
{
    static bool use_environment_map;
    static bool use_grid;

    // Editors

    bool skip_tutorial = false;

    EditorType current_editor_type;

    std::vector<BaseEditor*> editors;

    BaseEditor* current_editor = nullptr;

    std::vector<Node*> to_delete;

    // Animation System

    Skeleton default_skeleton;

    void init_default_skeleton();

    // UI

    ui::ContextMenu* active_context_menu = nullptr;

    void render_gui();

    // Engine meshes
    Environment3D* environment = nullptr;
    MeshInstance3D* grid = nullptr;
    MeshInstance3D* controller_mesh_left = nullptr;
    MeshInstance3D* controller_mesh_right = nullptr;
    MeshInstance3D* ray_pointer = nullptr;
    MeshInstance3D* sphere_pointer = nullptr;

    Gizmo3D gizmo;

    // make Cursor all static and remove this??
    ui::Cursor cursor;

public:

    int initialize(Renderer* renderer, sEngineConfiguration configuration = {}) override;
    virtual int post_initialize() override;

    void clean() override;

	void update(float delta_time) override;
	void render() override;

    void resize_window(int width, int height) override;

    void set_main_scene(const std::string& scene_path);
    void add_to_main_scene(const std::string& scene_path);

    static void render_controllers();

    static void switch_editor(uint8_t editor_idx, void* data = nullptr);
    static void toggle_use_grid();
    static void toggle_use_environment_map();

    void set_current_sculpt(SculptNode* sculpt_instance);

    void push_context_menu(ui::ContextMenu* cm);
    void delete_context_menu(ui::ContextMenu* cm);

    inline Gizmo3D* get_gizmo() { return &gizmo; }
    inline BaseEditor* get_current_editor() const { return current_editor; }
    inline EditorType get_current_editor_type() const { return current_editor_type; }

    Skeleton* get_default_skeleton() { return &default_skeleton; }

    template <typename T = BaseEditor*>
    inline T get_editor(uint8_t editor_idx) const { return static_cast<T>(editors[editor_idx]); }
};
