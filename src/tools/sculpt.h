#pragma once

#include "tool.h"
#include "graphics/edit.h"

#include <glm/glm.hpp>


class SculptTool : EditorTool {
	sEdit edit_to_add;

	// Modifiers
	// Mirror
	bool use_mirror = false;
	glm::vec3 mirror_origin = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 mirror_normal = glm::vec3(0.0f, 0.0f, 1.0f);

	// Aditional Sausage primitive
	bool is_sausage_start_setted = false;
	glm::vec3 sausage_start_position = glm::vec3(0.0f, 0.0f, 0.0f);

public:
	void initialize();
	void clean();

	std::vector<sEdit> use();
	void update(float delta_tim);
	void render_scene();
	void render_ui();
};