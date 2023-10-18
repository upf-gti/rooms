#pragma once

#include "engine.h"
#include "tools/sculpt/sculpt_editor.h"

class RoomsEngine : public Engine {

    std::vector<Entity*> entities;

    SculptEditor sculpt_editor;

    bool export_scene();
    bool import_scene();

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;
};
