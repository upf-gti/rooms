#pragma once

#include "engine.h"
#include "tools/tool.h"
#include "ui/ui_controller.h"

enum eTool : uint8_t {
	NONE = 0,
	SCULPTING,
	COLOR,
	TOOL_COUNT
};

class RoomsEngine : public Engine {

	std::vector<Entity*>	entities;

	EditorTool*				tools[TOOL_COUNT];

	eTool					current_tool = NONE;
	ui::Controller			tool_controller;

	void enable_tool( eTool tool );

public:

	virtual int initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen) override;

	virtual void update(float delta_time) override;
	virtual void render() override;
};
