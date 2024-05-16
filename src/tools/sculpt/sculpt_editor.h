#pragma once

#include "graphics/edit.h"
#include "framework/nodes/ui.h"
#include "framework/nodes/text.h"
#include "framework/ui/gizmo_3d.h"

#include <map>

enum eTool : uint8_t {
    NONE = 0,
    SCULPT,
    PAINT,
    TOOL_COUNT
};

enum : uint8_t {
    LAYOUT_NONE = 1 << 0,
    LAYOUT_SCULPT = 1 << 1,
    LAYOUT_PAINT = 1 << 2,
    LAYOUT_SHIFT_R = 1 << 3,
    LAYOUT_SHIFT_L = 1 << 4,
    LAYOUT_NO_SHIFT_R = 1 << 5,
    LAYOUT_NO_SHIFT_L = 1 << 6,
    LAYOUT_SCULPT_SHIFT_R = LAYOUT_SCULPT | LAYOUT_SHIFT_R,
    LAYOUT_SCULPT_SHIFT_L = LAYOUT_SCULPT | LAYOUT_SHIFT_L,
    LAYOUT_SCULPT_NO_SHIFT_R = LAYOUT_SCULPT | LAYOUT_NO_SHIFT_R,
    LAYOUT_SCULPT_NO_SHIFT_L = LAYOUT_SCULPT | LAYOUT_NO_SHIFT_L,
    LAYOUT_PAINT_SHIFT_R = LAYOUT_PAINT | LAYOUT_SHIFT_R,
    LAYOUT_PAINT_SHIFT_L = LAYOUT_PAINT | LAYOUT_SHIFT_L,
    LAYOUT_PAINT_NO_SHIFT_R = LAYOUT_PAINT | LAYOUT_NO_SHIFT_R,
    LAYOUT_PAINT_NO_SHIFT_L = LAYOUT_PAINT | LAYOUT_NO_SHIFT_L,
    LAYOUT_ANY_SHIFT_R = LAYOUT_SCULPT_SHIFT_R | LAYOUT_PAINT_SHIFT_R,
    LAYOUT_ANY_SHIFT_L = LAYOUT_SCULPT_SHIFT_L | LAYOUT_PAINT_SHIFT_L,
    LAYOUT_ANY_NO_SHIFT_R = LAYOUT_SCULPT_NO_SHIFT_R | LAYOUT_PAINT_NO_SHIFT_R,
    LAYOUT_ANY_NO_SHIFT_L = LAYOUT_SCULPT_NO_SHIFT_L | LAYOUT_PAINT_NO_SHIFT_L,
    LAYOUT_SCULPT_ALL = (LAYOUT_SCULPT_SHIFT_R | LAYOUT_SCULPT_SHIFT_L) | (LAYOUT_SCULPT_NO_SHIFT_R | LAYOUT_SCULPT_NO_SHIFT_L),
    LAYOUT_PAINT_ALL = (LAYOUT_PAINT_SHIFT_R | LAYOUT_PAINT_SHIFT_L) | (LAYOUT_PAINT_NO_SHIFT_R | LAYOUT_PAINT_NO_SHIFT_L),
    LAYOUT_ALL = LAYOUT_SCULPT_ALL | LAYOUT_PAINT_ALL
};

class RoomsRenderer;
class MeshInstance3D;
class Viewport3D;

struct PrimitiveState {
    glm::vec4 dimensions;
    // color?
    // modifiers?
};

class SculptEditor {

    RoomsRenderer*  renderer = nullptr;

    MeshInstance3D* sculpt_area_box = nullptr;

    bool sculpt_started = false;
    bool was_tool_used = false;

    eTool current_tool = eTool::NONE;

    bool ui_edit_to_add = false;

    static uint8_t last_generated_material_uid;
    uint8_t num_generated_materials = 0u;

    bool is_rotation_being_used();
    void add_pbr_material_data(const std::string& name, const Color& base_color, float roughness, float metallic,
        float noise_intensity = 0.0f, const Color& noise_color = colors::RUST, float noise_frequency = 20.0f, int noise_octaves = 8);
    void generate_material_from_stroke(void* button);
    void generate_random_material();
    void update_stroke_from_material(const std::string& name);
    void update_gui_from_stroke_material(const StrokeMaterial& mat);
    void pick_material();

    /*
    *	Edits
    */

    std::vector<Edit>   preview_tmp_edits;
    std::vector<Edit>   new_edits;

    std::map<uint32_t, PrimitiveState> primitive_default_states = {};

    std::map<std::string, PBRMaterialData> pbr_materials_data = {};

    Edit             edit_to_add = {};
    StrokeParameters stroke_parameters = {};

    MeshInstance3D* mesh_preview = nullptr;
    MeshInstance3D* mesh_preview_outline = nullptr;

    void set_primitive(sdPrimitive primitive);
    void set_operation(sdOperation operation);

    void set_edit_size(float main = -1.0f, float secondary = -1.0f, float round = -1.0f);

    void update_edit_preview(const glm::vec4& dims);

    void set_onion_modifier(float value);
    void set_cap_modifier(float value);

    bool must_render_mesh_preview_outline();

    bool can_snap_to_surface();

    bool dimensions_dirty       = true;
    bool modifiers_dirty        = false;
    bool stamp_enabled          = false;
    bool rotation_started       = false;
    bool snap_to_surface        = false;
    bool is_picking_material    = false;
    bool was_material_picked    = false;

    glm::vec3 sculpt_start_position = {};
    glm::vec3 edit_position_world = {};
    glm::vec3 initial_hand_translation = {};
    glm::vec3 translation_diff = {};

    struct {
        glm::vec3 prev_controller_pos = {};
        glm::vec3 controller_velocity = {};
        glm::vec3 controller_acceleration = {};
        glm::vec3 controller_frame_distance = {};
        glm::vec3 prev_edit_position = {};
    } controller_position_data;

    glm::quat initial_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat sculpt_rotation = { 0.0, 0.0, 0.0, 1.0 };
    glm::quat edit_rotation_world = {};

    float hand_to_edit_distance = 0.0f;

    /*
    *	Modifiers
    */

    // Shape Editor

    float onion_thickness = 0.0f;
    float capped_value = -1.0f; // -1..1 no cap..fully capped

    // Axis lock

    enum : uint8_t {
        AXIS_LOCK_X = 1 << 0,
        AXIS_LOCK_Y = 1 << 1,
        AXIS_LOCK_Z = 1 << 2,
    };

    bool axis_lock = false;
    Gizmo3D axis_lock_gizmo = {};
    uint8_t axis_lock_mode = AXIS_LOCK_Z;
    glm::vec3 axis_lock_position = glm::vec3(0.f);

    // Snap to grid

    bool snap_to_grid = false;
    float snap_grid_size = 0.05f;

    // Mirror

    bool use_mirror = false;

    Gizmo3D mirror_gizmo = {};
    MeshInstance3D* mirror_mesh = nullptr;

    glm::vec3 mirror_origin = glm::vec3(0.f);
    glm::vec3 mirror_normal = glm::vec3(0.f, 0.f, 1.f);
    glm::quat mirror_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };

    // Controller UI
    ui::VContainer2D* right_hand_container = nullptr;
    ui::VContainer2D* left_hand_container = nullptr;
    Viewport3D* right_hand_ui_3D = nullptr;
    Viewport3D* left_hand_ui_3D = nullptr;

    // Meta quest controller meshes
    MeshInstance3D* controller_mesh_left = nullptr;
    MeshInstance3D* controller_mesh_right = nullptr;
    
    // Main pannel UI
    ui::HContainer2D* main_panel_2d = nullptr;
    Viewport3D* main_panel_3d = nullptr;

    size_t max_recent_colors = 0;
    std::vector<Color> recent_colors;

    void init_ui();
    void bind_events();
    void add_recent_color(const Color& color);

    // Stamp slide
    glm::vec3 edit_position_stamp = {};
    glm::vec3 edit_origin_stamp = {};
    glm::quat edit_rotation_stamp = { 0.0f, 0.0f, 0.0f, 1.0f };

    // Editor
    bool is_tool_pressed = false;
    bool is_released = false;
    bool was_tool_pressed = false;
    bool is_stretching_edit = false;

    bool is_shift_left_pressed = false;
    bool is_shift_right_pressed = false;

    bool edit_update(float delta_time);
    void mirror_current_edits(float delta_time);
    void apply_mirror_position(glm::vec3& position);
    void apply_mirror_rotation(glm::quat& position) const;

    glm::vec3 world_to_texture3d(const glm::vec3& position, bool skip_translation = false);
    glm::vec3 texture3d_to_world(const glm::vec3& position);
    void scene_update_rotation();

public:

    SculptEditor() {};

    void initialize();
    void clean();

    void update(float delta_time);
    void render();
    void render_gui();

    void enable_tool(eTool tool);
    void set_sculpt_started(bool value);

    bool is_tool_being_used(bool stamp_enabled);

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
