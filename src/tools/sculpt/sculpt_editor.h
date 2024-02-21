#pragma once

#include <graphics/edit.h>

#include "framework/ui/transform_gizmo.h"
#include "framework/ui/ui_controller.h"

enum eTool : uint8_t {
    NONE = 0,
    SCULPT,
    PAINT,
    SWEEP,
    TOOL_COUNT
};

class RoomsRenderer;

struct PrimitiveState {
    glm::vec4 dimensions;
    // color?
    // modifiers?
};

class SculptEditor {

    RoomsRenderer*  renderer = nullptr;
    EntityMesh*     floor_grid_mesh = nullptr;

    bool sculpt_started = false;
    bool was_tool_used = false;

    eTool current_tool = NONE;

    static uint8_t last_generated_material_uid;
    uint8_t num_generated_materials = 0u;

    bool is_rotation_being_used();
    void add_pbr_material_data(const std::string& name, const Color& base_color, float roughness, float metallic,
        float noise_intensity = 0.0f, const Color& noise_color = colors::RUST, float noise_frequency = 20.0f, int noise_octaves = 8);
    void generate_material_from_stroke(void* button);
    void update_stroke_from_material(const std::string& name);
    void update_gui_from_stroke_material(const StrokeMaterial& mat);
    void pick_material();

    /*
    *	Edits
    */

    std::vector<Edit>   preview_tmp_edits;
    std::vector<Edit>   new_edits;

    std::map<uint32_t, PrimitiveState> primitive_default_states;

    std::map<std::string, PBRMaterialData> pbr_materials_data;

    Edit             edit_to_add;
    StrokeParameters stroke_parameters;

    EntityMesh*     mesh_preview = nullptr;
    EntityMesh*     mesh_preview_outline = nullptr;

    void set_primitive( sdPrimitive primitive );
    void update_edit_preview( const glm::vec4& dims );

    void set_onion_modifier(float value);
    void set_cap_modifier(float value);

    void toggle_capped_modifier();
    void toggle_onion_modifier();

    bool mustRenderMeshPreviewOutline();

    bool canSnapToSurface();

    bool        modifiers_dirty     = false;
    bool        dimensions_dirty    = true;
    bool        stamp_enabled       = false;
    bool		rotation_started    = false;
    bool        snap_to_surface     = false;
    bool        is_picking_material = false;
    bool        was_material_picked = false;

    glm::vec3	sculpt_start_position;
    glm::vec3	edit_position_world;
    glm::vec3	initial_hand_translation = {};
    glm::vec3	translation_diff = {};

    glm::quat	initial_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat	rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat	sculpt_rotation = { 0.0, 0.0, 0.0, 1.0 };

    float       hand2edit_distance = 0.2f;

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
    glm::vec3 mirror_normal = glm::vec3(0.f, 0.f, 1.f);

    
    // UI

    ui::Controller        gui;
    ui::Controller        helper_gui;
    size_t                max_recent_colors;
    std::vector<Color>    recent_colors;

    void bind_events();
    void add_recent_color(const Color& color);

    // Editor

    bool is_tool_being_used(bool stamp_enabled);
    bool edit_update(float delta_time);
    void mirror_current_edits(float delta_time);
    void mirror_position(glm::vec3& position);

    glm::vec3 world_to_texture3d(const glm::vec3& position);
    glm::vec3 texture3d_to_world(const glm::vec3& position);
    void scene_update_rotation();

public:

    void initialize();
    void clean();

    void update(float delta_time);
    void render();
    void render_gui();

    void enable_tool(eTool tool);
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
