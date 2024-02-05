#pragma once

#include "engine.h"
#include "tools/sculpt/sculpt_editor.h"

#include <vector>

class Entity;

class RoomsEngine : public Engine {

    static std::vector<Entity*> entities;
    static EntityMesh* skybox;

    SculptEditor sculpt_editor;

    bool export_scene();
    bool import_scene();

    void render_gui();
    bool show_tree_recursive(Entity* entity);

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;
};
