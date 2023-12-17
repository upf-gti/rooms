#pragma once

#include "graphics/edit.h"
#include "framework/input.h"

class Tool {

protected:

	bool enabled = false;

    Edit edit_to_add = {
        .position = glm::vec3(0.0f, 0.0f, 0.0f),
        .dimensions = glm::vec4(0.01f, 0.01f, 0.01f, 0.f)
    };

    Edit previous_edit;

	// Timestepping counters
	float edit_update_counter = 0.0f;

    bool is_tool_activated() {
#ifdef XR_SUPPORT
        return Input::was_key_pressed(GLFW_KEY_SPACE) ||
            (stamp ? Input::was_trigger_pressed(HAND_RIGHT) : Input::get_trigger_value(HAND_RIGHT) > 0.5f);
#else
        return Input::is_key_pressed(GLFW_KEY_SPACE);
#endif
    }

public:

    bool stamp = false;

    virtual void initialize();
    virtual void clean();

	virtual void start() { enabled = true; }
	virtual void stop() { enabled = false; }

    virtual bool use_tool();

    virtual bool update(float delta_time, StrokeParameters& stroke_parameters);
	virtual void render_scene() {}
	virtual void render_ui() {}

    Edit& get_edit_to_add() { return edit_to_add; }

};
