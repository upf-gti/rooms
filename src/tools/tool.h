#pragma once

#include <vector>
#include "graphics/edit.h"
#include "framework/input.h"

class EditorTool {
public:

	bool enabled = false;
	// Timestepping counters
	float  edit_update_counter = 0.0f;

	virtual void initialize() {}
	virtual void clean() {}

	virtual void start() { enabled = true; }
	virtual void stop() { enabled = false; }

	virtual bool use_tool() {
		if (edit_update_counter > 0.016f) {
			edit_update_counter -= 0.016f;
			return true;
		}
		
		return false;
	}

	virtual void update(float delta_time) {
		edit_update_counter += delta_time;
	}

	virtual void render_scene() {}
	virtual void render_ui() {}

};