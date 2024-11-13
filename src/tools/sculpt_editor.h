#pragma once

#include "tools/base_editor.h"

#include "graphics/edit.h"

#include "framework/nodes/text.h"
#include "framework/ui/gizmo_3d.h"
#include "framework/math/spline.h"

#include "graphics/managers/stroke_manager.h"
#include "graphics/managers/sculpt_manager.h"

#include <map>

enum eTool : uint8_t {
    NONE = 0,
    SCULPT,
    PAINT,
    TOOL_COUNT
};

class MeshInstance3D;
class SculptNode;

struct PrimitiveState {
    glm::vec4 dimensions;
    glm::quat rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    // modifiers?
};

class SculptEditor : public BaseEditor {

    MeshInstance3D* sculpt_area_box = nullptr;

    SculptNode* current_sculpt = nullptr;

    StrokeManager   stroke_manager = {};

    bool sculpt_started = false;
    bool was_tool_used = false;

    bool called_undo = false;
    bool called_redo = false;

    eTool current_tool = eTool::NONE;

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

    void set_edit_size(float main = -1.0f, float secondary = -1.0f, float round = -1.0f);
    void set_primitive(sdPrimitive primitive);
    void set_operation(sdOperation operation);
    void set_onion_modifier(float value);
    void set_cap_modifier(float value);

    void update_edit_preview(const glm::vec4& dims);
    bool must_render_mesh_preview_outline();
    bool can_snap_to_surface();

    bool dimensions_dirty       = true;
    bool force_new_stroke       = false;
    bool stamp_enabled          = false;
    bool rotation_started       = false;
    bool edit_rotation_started  = false;
    bool snap_to_surface        = false;
    bool is_picking_material    = false;
    bool was_material_picked    = false;

    glm::quat last_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec3 last_hand_translation = {};
    glm::vec3 last_snap_position = {};

    // Intersections
    glm::vec3 ray_origin;
    glm::vec3 ray_direction;
    sGPU_SculptResults last_gpu_results;

    // Edit
    glm::quat edit_rotation_diff = { 0.0, 0.0, 0.0, 1.0 };
    glm::quat edit_user_rotation = { 0.0, 0.0, 0.0, 1.0 };
    glm::quat edit_rotation_world = {};
    glm::vec3 edit_position_world = {};

    // Stamp slide
    glm::vec3 edit_position_stamp = {};
    glm::vec3 edit_origin_stamp = {};
    glm::quat edit_rotation_stamp = { 0.0f, 0.0f, 0.0f, 1.0f };

    float hand_to_edit_distance = 0.0f;

    // Axis lock

    enum : uint8_t {
        AXIS_LOCK_X = 1 << 0,
        AXIS_LOCK_Y = 1 << 1,
        AXIS_LOCK_Z = 1 << 2,
    };

    bool axis_lock = false;
    Gizmo3D axis_lock_gizmo = {};
    uint8_t axis_lock_mode = AXIS_LOCK_Z;

    // Snap to grid

    bool snap_to_grid = false;
    float snap_grid_size = 0.05f;

    // Mirror

    bool use_mirror = false;

    Gizmo3D mirror_gizmo = {};
    MeshInstance3D* mirror_mesh = nullptr;
    glm::vec3 mirror_normal = glm::vec3(0.f, 0.f, 1.f);

    /*
    *	UI
    */

    size_t max_recent_colors = 0;
    std::vector<Color> recent_colors;

    void init_ui();
    void bind_events();
    void add_recent_color(const Color& color);
    void generate_shortcuts() override;

    /*
    *	Splines
    */

    bool creating_spline = false;
    bool dirty_spline = false;
    BezierSpline current_spline;
    BezierSpline preview_spline;

    void start_spline();
    void reset_spline(bool update_ui = true);
    void end_spline();

    /*
    *	Editor
    */

    bool is_tool_pressed = false;
    bool is_released = false;
    bool was_tool_pressed = false;
    bool is_stretching_edit = false;

    bool edit_update(float delta_time);
    void mirror_current_edits(float delta_time);
    void apply_mirror_position(glm::vec3& position);
    void apply_mirror_rotation(glm::quat& position) const;

    glm::vec3 world_to_texture3d(const glm::vec3& position, bool skip_translation = false);
    glm::vec3 texture3d_to_world(const glm::vec3& position);

    void update_sculpt_rotation();
    void update_edit_rotation();

    void undo();
    void redo();

public:

    SculptEditor() {};
    SculptEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void enable_tool(eTool tool);
    void set_sculpt_started(bool value);

    void set_preview_edits(const std::vector<Edit> &edit_previews);

    void set_current_sculpt(SculptNode* sculpt_instance);
    SculptNode* get_current_sculpt() { return current_sculpt; }

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

    void on_enter(void* data);
    // void on_exit();
};
