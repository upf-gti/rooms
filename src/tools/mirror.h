#pragma once

#include "tool.h"

class MirrorTool : EditorTool {
	glm::vec3 mirror_origin = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 mirror_normal = glm::vec3(0.0f, 1.0f, 0.0f);

public:
	void initialize();
	void clean();

	std::vector<sEdit> use();
	void update(float delta_tim);
	void render_scene();
	void render_ui();
};