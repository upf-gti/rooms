#pragma once

#include "graphics/edit.h"
#include "framework/input.h"
#include "ui/ui_controller.h"

class RaymarchingRenderer;

class Tool {

protected:

	bool enabled = false;

    Edit edit_to_add = {
        .position = glm::vec3(0.0f, 0.0f, 0.0f),
        .primitive = SD_SPHERE,
        .color = glm::vec3(1.0f, 0.0f, 0.0f),
        .operation = OP_SMOOTH_UNION,
        .dimensions = glm::vec4(0.01f, 0.01f, 0.01f, 0.f)
    };

    Edit previous_edit;

	// Timestepping counters
	float edit_update_counter = 0.0f;

    bool is_tool_activated() {
#ifdef XR_SUPPORT
        
        return Input::is_key_pressed(GLFW_KEY_A) || (Input::was_button_pressed(XR_BUTTON_B) || (Input::get_trigger_value(HAND_RIGHT) > 0.5));
#else
        return Input::is_key_pressed(GLFW_KEY_A);
#endif
    }

public:
    virtual void initialize();
    virtual void clean();

	virtual void start() { enabled = true; }
	virtual void stop() { enabled = false; }

    virtual bool use_tool();

    virtual bool update(float delta_time);

	virtual void render_scene() {}
	virtual void render_ui() {}

    Edit& get_edit_to_add() {
        return edit_to_add;
    }
};
