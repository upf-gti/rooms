#pragma once

#include "engine/engine.h"

#include "framework/ui/gizmo_3d.h"
#include "framework/ui/cursor.h"

class Node;
class Node3D;
class SculptNode;
class Environment3D;
class Stage;

enum StageType : uint8_t {
    SCENE_EDITOR,
    SCULPT_EDITOR,
    ANIMATION_EDITOR,
    TUTORIAL_STAGE,
    PLAYER_STAGE,
    MENU_STAGE,
    STAGE_COUNT
};

class RoomsEngine : public Engine
{
    static bool use_environment_map;
    static bool use_grid;

    // Stages

    bool skip_tutorial = false;

    StageType current_stage_type;

    std::vector<Stage*> stages;

    Stage* current_stage = nullptr;

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

    void set_main_scene(const std::string& scene_path);
    void add_to_main_scene(const std::string& scene_path);

    static void render_controllers();

    static void switch_stage(uint8_t stage_idx, void* data = nullptr);
    static void toggle_use_grid();
    static void toggle_use_environment_map();

    void set_current_sculpt(SculptNode* sculpt_instance);

    inline Gizmo3D* get_gizmo() { return &gizmo; }
    inline Stage* get_current_stage() const { return current_stage; }
    inline StageType get_current_stage_type() const { return current_stage_type; }

    template <typename T = Stage*>
    inline T get_stage(uint8_t stage_idx) const { return static_cast<T>(stages[stage_idx]); }
};
