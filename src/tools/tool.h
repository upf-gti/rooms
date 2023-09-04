#pragma once

#include <vector>
#include "graphics/edit.h"
#include "framework/input.h"

class EditorTool {
public:

	bool enabled = false;

	virtual void initialize() {}
	virtual void clean() {}

	virtual void start() { enabled = true; }
	virtual void stop() { enabled = false; }

	virtual void update(float delta_time) {}
	virtual void render_scene() {}
	virtual void render_ui() {}
};