#pragma once

#include "tool.h"

enum eTool : uint8_t {
    NONE = 0,
    SCULPT,
    PAINT,
    TOOL_COUNT
};

class SculptEditor {

    RaymarchingRenderer*    renderer = nullptr;
    bool                    sculpt_started = false;
    Tool*                   tools[TOOL_COUNT];
    eTool					current_tool = NONE;
    ui::Controller			gui;

    /*
    *	Edits
    */

    Color                   current_color = colors::RED;
    sdPrimitive             current_primitive = SD_SPHERE;

    EntityMesh*             mesh_preview = nullptr;

    bool			        rotation_started = false;

    glm::vec3		        sculpt_start_position;
    glm::vec3		        initial_hand_translation = {};
    glm::vec3		        translation_diff = {};

    glm::quat		        initial_hand_rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat		        rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
    glm::quat		        sculpt_rotation = { 0.0, 0.0, 0.0, 1.0 };

    /*
    *	Modifiers
    */

    // Mirror

    bool			use_mirror = false;
    glm::vec3		mirror_origin = glm::vec3(0.0f, -0.5f, 0.0f);
    glm::vec3		mirror_normal = glm::vec3(1.0f, 0.0f, 0.0f);

    void enable_tool(eTool tool);

    bool is_rotation_being_used() {
        return Input::get_trigger_value(HAND_LEFT) > 0.5;
    }

public:

    void initialize();
    void update(float delta_time);
    void render();
};
