#pragma once

#include <vector>
#include "graphics/edit.h"
#include "framework/input.h"

class EditorTool {
public:
	virtual void initialize() {}
	virtual void clean() {}

	virtual void update(float delta_time) {}
	virtual void render_scene() {}
	virtual void render_ui() {}
};