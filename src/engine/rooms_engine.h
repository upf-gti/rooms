#pragma once

#include "engine.h"
#include "tools/sculpt/sculpt_editor.h"
#include "ui/ui_controller.h"

class RoomsEngine : public Engine {

	std::vector<Entity*>	entities;

    SculptEditor            sculpt_editor;

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;
};
