#pragma once

#include <vector>
#include "graphics/edit.h"
#include "framework/input.h"

class EditorTool {
public:
	bool is_smooth					= true;
	sdPrimitive selected_primitive	= SD_SPHERE;
	glm::vec3 size					= glm::vec3(0.1f, 0.1f, 0.1f);
	glm::vec3 color					= glm::vec3(1.0f, 1.0f, 1.0f);
	float radius					= 0.1f;

	virtual void initialize();
	virtual void clean();

	virtual std::vector<sEdit> use();
	virtual void update(float delta_tim);
	virtual void render_scene();
	virtual void render_ui();
};