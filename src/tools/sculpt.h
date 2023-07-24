#pragma once

#include "tool.h"
#include "graphics/edit.h"
#include "graphics/raymarching_renderer.h"
#include "ui/ui.h"

#include <glm/glm.hpp>


class SculptTool : EditorTool {
	RaymarchingRenderer  *edit_renderer;
	ui::Controller  *ui_controller;


	sEdit edit_to_add = {
		.position = glm::vec3(0.0f, 0.0f, 0.0f),
		.primitive = SD_SPHERE,
		.color = glm::vec3(1.0f, 0.0f, 0.0f),
		.operation = OP_SMOOTH_UNION,
		.size = glm::vec3(0.1f, 0.1f, 0.1f),
		.radius = 0.05f
	};

	// Modifiers
	// Mirror
	bool use_mirror = true;
	glm::vec3 mirror_origin = glm::vec3(0.0f, -0.5f, 0.0f);
	glm::vec3 mirror_normal = glm::vec3(1.0f, 0.0f, 0.0f);

	// Aditional Sausage primitive
	bool is_sausage_start_setted = false;
	glm::vec3 sausage_start_position = glm::vec3(0.0f, 0.0f, 0.0f);

public:
	void initialize(Renderer* edit_render, ui::Controller *ui_controller);
	void clean();

	void update(float delta_tim);
	void render_scene();
	void render_ui();
};