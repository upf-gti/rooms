#pragma once

#include "engine.h"
#include "tools/tool.h"

enum eTool : uint8_t {
	ADDITION = 0,
	MIRROR,
	EDITOR_TOOL_COUNT
};

class RoomsEngine : public Engine {

	std::vector<Entity*> entities;

	EditorTool *tools[EDITOR_TOOL_COUNT];

public:

	virtual int initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen) override;

	virtual void update(float delta_time) override;
	virtual void render() override;
};
