#pragma once

#include "tool.h"

class SausageTool : EditorTool {
	glm::vec3 second_position = glm::vec3(0.0f, -1.0f, 0.0f);

public:
	void initialize();
	void clean();

	std::vector<sEdit> use();
	void update(float delta_tim);
	void render_scene();
	void render_ui();
};