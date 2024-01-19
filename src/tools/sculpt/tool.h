#pragma once

#include "graphics/edit.h"

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

    bool is_tool_activated();

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
