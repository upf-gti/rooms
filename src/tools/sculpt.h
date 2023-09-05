#pragma once

#include "tool.h"
#include "graphics/edit.h"
#include "ui/ui_controller.h"
#include <glm/glm.hpp>

class RaymarchingRenderer;

class SculptTool : EditorTool {

	RaymarchingRenderer*	renderer = nullptr;
	EntityMesh*				mesh_preview = nullptr;
	ui::Controller			ui_controller;

	sEdit					edit_to_add = {
								.position = glm::vec3(0.0f, 0.0f, 0.0f),
								.primitive = SD_BOX,
								.color = glm::vec3(1.0f, 0.0f, 0.0f),
								.operation = OP_SMOOTH_UNION,
								.size = glm::vec3(0.1f, 0.1f, 0.1f),
								.radius = 0.01f
							};


	bool			sculpt_started = false;

	/*
	*	Modifiers
	*/

	// Mirror

	bool			use_mirror = false;
	glm::vec3		mirror_origin = glm::vec3(0.0f, -0.5f, 0.0f);
	glm::vec3		mirror_normal = glm::vec3(1.0f, 0.0f, 0.0f);

	// Aditional Sausage primitive

	bool			is_sausage_start_setted = false;
	glm::vec3		sausage_start_position = glm::vec3(0.0f, 0.0f, 0.0f);
	bool			has_trigger_used = false;

public:

	void initialize();
	void clean();

	void update(float delta_time);
	void render_scene();
	void render_ui();
};