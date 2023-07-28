#pragma once

#include "engine.h"
#include "tools/tool.h"

#include "ui/ui_controller.h"

enum eTool : uint8_t {
	SCULPTING = 0,
	COLOR,
	EDITOR_TOOL_COUNT
};

class RoomsEngine : public Engine {

	std::vector<Entity*>	entities;

	EditorTool				*tools[EDITOR_TOOL_COUNT];

	eTool					current_tool = SCULPTING;

	EntityMesh				*right_controller_mesh = NULL;
	EntityMesh				*left_controller_mesh = NULL;

	ui::Controller			tool_selection_ui_controller;

public:

	virtual int initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen) override;

	virtual void update(float delta_time) override;
	virtual void render() override;
};
