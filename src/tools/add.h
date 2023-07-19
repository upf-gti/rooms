#pragma once

#include "tool.h"

class AdditionTool : EditorTool {

public:
	void initialize();
	void clean();

	std::vector<sEdit> use();
	void update(float delta_tim);
	void render_scene();
	void render_ui();
};