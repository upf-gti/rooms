#pragma once

#include <string>
#include <unordered_map>

#include "framework/input_xr.h"

class RoomsRenderer;
class Node2D;
class Gizmo3D;

namespace ui {
    class HContainer2D;
    class VContainer2D;
}

namespace shortcuts {
    // Left hand
    const std::string X_BUTTON_PATH = "data/textures/buttons/x.png";
    const std::string Y_BUTTON_PATH = "data/textures/buttons/y.png";
    const std::string L_GRIP_X_BUTTON_PATH = "data/textures/buttons/l_grip_plus_x.png";
    const std::string L_GRIP_Y_BUTTON_PATH = "data/textures/buttons/l_grip_plus_y.png";
    const std::string L_TRIGGER_PATH = "data/textures/buttons/l_trigger.png";
    const std::string L_GRIP_L_TRIGGER_PATH = "data/textures/buttons/l_grip_plus_l_trigger.png";
    const std::string L_THUMBSTICK_PATH = "data/textures/buttons/l_thumbstick.png";
    const std::string L_GRIP_L_THUMBSTICK_PATH = "data/textures/buttons/l_grip_plus_l_thumbstick.png";

    // Right hand
    const std::string A_BUTTON_PATH = "data/textures/buttons/a.png";
    const std::string B_BUTTON_PATH = "data/textures/buttons/b.png";
    const std::string R_GRIP_A_BUTTON_PATH = "data/textures/buttons/r_grip_plus_a.png";
    const std::string R_GRIP_B_BUTTON_PATH = "data/textures/buttons/r_grip_plus_b.png";
    const std::string R_TRIGGER_PATH = "data/textures/buttons/r_trigger.png";
    const std::string R_GRIP_R_TRIGGER_PATH = "data/textures/buttons/r_grip_plus_r_trigger.png";
    const std::string R_THUMBSTICK_PATH = "data/textures/buttons/r_thumbstick.png";
    const std::string R_GRIP_R_THUMBSTICK_PATH = "data/textures/buttons/r_grip_plus_r_thumbstick.png";

    enum : uint8_t {
        TOGGLE_SCENE_INSPECTOR,
        EDIT_SCULPT_NODE,
        EDIT_GROUP,
        ANIMATE_NODE,
        SELECT_NODE,
        DUPLICATE_NODE,
        CLONE_NODE,
        PLACE_NODE,
        GROUP_NODE,
        ADD_TO_GROUP,
        CREATE_GROUP,
        UNGROUP,
        SCENE_UNDO,
        SCENE_REDO,
        BACK_TO_SCENE,
        ROUND_SHAPE,
        MODIFY_SMOOTH,
        REDO,
        UNDO,
        OPEN_GUIDES_MENU,
        OPEN_PBR_MENU,
        MANIPULATE_SCULPT,
        MAIN_SIZE,
        SECONDARY_SIZE,
        ADD_SUBSTRACT,
        CONFIRM_SPLINE,
        CANCEL_SPLINE,
        ADD_KNOT,
        SCULPT_PAINT,
        PICK_MATERIAL,
        STAMP,
        SMEAR
    };
}

enum eTutorialStep {
    TUTORIAL_NONE,
    TUTORIAL_WELCOME,
    TUTORIAL_ROOM,
    TUTORIAL_SCENE_INSPECTOR,
    TUTORIAL_ADD_NODE,
    TUTORIAL_EDIT_SCULPT,
    TUTORIAL_STAMP_SMEAR,
    TUTORIAL_PRIMITIVES_OPERATIONS,
    TUTORIAL_CURVES,
    TUTORIAL_GUIDES,
    TUTORIAL_MATERIAL,
    TUTORIAL_PAINT,
    TUTORIAL_UNDO_REDO,
    TUTORIAL_CLOSE,
    TUTORIAL_PANEL_COUNT,
};

enum InspectNodeFlags {
    NODE_NAME = 1 << 0,
    NODE_ICON = 1 << 1,
    NODE_VISIBILITY = 1 << 2,
    NODE_EDIT = 1 << 3,
    NODE_DELETE = 1 << 4,
    NODE_ANIMATE = 1 << 5,
    NODE_CHILDREN = 1 << 6,
    NODE_SUBMENU_ICON = 1 << 7,
    NODE_STANDARD = NODE_NAME | NODE_VISIBILITY | NODE_ANIMATE | NODE_DELETE,
    NODE_LIGHT = NODE_STANDARD,
    NODE_SCULPT = NODE_STANDARD | NODE_EDIT,
    NODE_GROUP = NODE_STANDARD | NODE_EDIT | NODE_CHILDREN
};

class BaseEditor {

    std::string name;

protected:

    bool is_shift_left_pressed = false;
    bool is_shift_right_pressed = false;

    float last_hover_time = 0.0f;

    struct sControllerMovementData {
        glm::vec3 prev_position = {};
        glm::vec3 velocity = {};
        glm::vec3 acceleration = {};
        glm::vec3 frame_distance = {};
        glm::vec3 prev_edit_position = {};
    };

    sControllerMovementData controller_movement_data[HAND_COUNT];

    RoomsRenderer* renderer = nullptr;

    Gizmo3D* gizmo = nullptr;

    // Controller UI
    ui::VContainer2D* right_hand_box = nullptr;
    ui::VContainer2D* left_hand_box = nullptr;

    // Main panel UI
    ui::HContainer2D* main_panel = nullptr;

    // Shortcuts
    virtual void generate_shortcuts() {}
    void update_shortcuts(const std::unordered_map<uint8_t, bool>& active_shortcuts);

    bool is_something_hovered();
    bool is_something_focused();

public:

    BaseEditor() {};
    BaseEditor(const std::string& name) : name(name) {};

    virtual void initialize() {};
    virtual void clean();

    virtual void update(float delta_time);
    virtual void render();
    virtual void render_gui() {};

    virtual void on_enter(void* data) {};
    virtual void on_exit() {};

    void set_name(const std::string& new_name) { name = new_name; }
    const std::string& get_name() { return name; }
};
