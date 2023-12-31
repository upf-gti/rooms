#pragma once

#include "tool.h"
#include "framework/ui/transform_gizmo.h"
#include "json_utils.h"
#include "framework/ui/ui_controller.h"

enum eTool : uint8_t {
    NONE = 0,
    SCULPT,
    PAINT,
    SWEEP,
    TOOL_COUNT
};

class RoomsRenderer;

class SculptEditor {

    RoomsRenderer*          renderer = nullptr;
    bool                    sculpt_started = false;
    bool                    was_tool_used = false;
    Tool*                   tools[TOOL_COUNT];
    eTool					current_tool = NONE;

    EntityMesh*             floor_grid_mesh = nullptr;

    std::vector<Edit>       preview_tmp_edits;
    std::vector<Edit>       new_edits;

    StrokeParameters        stroke_parameters;

    /*
    *	Edits
    */

    EntityMesh* mesh_preview = nullptr;
    EntityMesh* mesh_preview_outline = nullptr;


    void set_primitive( sdPrimitive primitive );
    void toggle_onion_modifier();
    void toggle_capped_modifier();
    void update_edit_preview( const glm::vec4& dims );

    bool        dimensions_dirty = true;
    bool        stamp_enabled = false;
    bool		rotation_started = false;

    glm::vec3	sculpt_start_position;
    glm::vec3	initial_hand_translation = {};
    glm::vec3	translation_diff = {};

    glm::quat	initial_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat	rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat	sculpt_rotation = { 0.0, 0.0, 0.0, 1.0 };

    /*
    *	Modifiers
    */

    // Shape Editor

    bool onion_enabled      = false;
    float onion_thickness   = 0.0f;

    bool capped_enabled = false;
    float capped_value  = -1.0f; // -1..1 no cap..fully capped

    // Axis lock

    enum : uint8_t {
        AXIS_LOCK_X = 1 << 0,
        AXIS_LOCK_Y = 1 << 1,
        AXIS_LOCK_Z = 1 << 2,
    };

    bool axis_lock = false;
    TransformGizmo axis_lock_gizmo;
    uint8_t axis_lock_mode = AXIS_LOCK_Z;
    glm::vec3 axis_lock_position = glm::vec3(0.f);

    // Snap to grid

    bool snap_to_grid = false;
    float snap_grid_size = 0.05f;

    // Mirror

    bool use_mirror = false;

    TransformGizmo mirror_gizmo;
    EntityMesh* mirror_mesh = nullptr;

    glm::vec3 mirror_origin = glm::vec3(0.f);
    glm::vec3 mirror_normal = glm::vec3(1.f, 0.f, 0.f);

    
    // UI

    ui::Controller        gui;
    ui::Controller        helper_gui;
    size_t                max_recent_colors;
    std::vector<Color>    recent_colors;

    void add_recent_color(const Color& color);

    void enable_tool(eTool tool);

    bool is_rotation_being_used() {
        return Input::get_trigger_value(HAND_LEFT) > 0.5;
    }

public:

    void initialize();
    void clean();
    void update(float delta_time);
    void render();

    void set_sculpt_started(bool value);

    void add_preview_edit_list(std::vector<Edit>& new_edit_lists) {
        for (Edit& edit : new_edit_lists) {
            preview_tmp_edits.push_back(edit);
        }
    }

    void add_edit_list(std::vector<Edit>& new_edit_lists) {
        for (Edit& edit : new_edit_lists) {
            new_edits.push_back(edit);
        }
    }
};
