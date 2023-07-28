#pragma once

#include "tool.h"
#include "graphics/edit.h"
#include "ui/ui_controller.h"
#include <glm/glm.hpp>

class RaymarchingRenderer;

class ColoringTool : EditorTool {

	RaymarchingRenderer*	renderer = nullptr;
	ui::Controller			ui_controller;

	sEdit					edit_to_add = {
								.position = glm::vec3(0.0f, 0.0f, 0.0f),
								.primitive = SD_SPHERE,
								.color = glm::vec3(1.0f, 0.0f, 0.0f),
								.operation = OP_PAINT,
								.size = glm::vec3(0.1f, 0.1f, 0.1f),
								.radius = 0.01f
							};

public:

	void initialize();
	void clean();

	void update(float delta_time);
	void render_scene();
	void render_ui();
};