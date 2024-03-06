#pragma once

#include "engine.h"
#include "tools/sculpt/sculpt_editor.h"

#include <vector>

class Node;
class Node3D;
class Environment3D;

class RoomsEngine : public Engine {

    static std::vector<Node3D*> entities;

    Environment3D* skybox = nullptr;

    SculptEditor sculpt_editor;

    bool export_scene();
    bool import_scene();

    void render_gui();
    bool show_tree_recursive(Node* entity);

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;
};
