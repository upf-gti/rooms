#pragma once

#include "engine/engine.h"

class Node;
class Node3D;
class MeshInstance3D;
class Environment3D;
class SculptEditor;

#define _DESTROY_(x) if(x) { delete x; }

class RoomsEngine : public Engine
{
    MeshInstance3D* raycast_pointer = nullptr;

    SculptEditor* sculpt_editor = nullptr;

    bool export_scene();
    bool import_scene();

    void render_gui();

public:

	int initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen) override;
    void clean() override;

	void update(float delta_time) override;
	void render() override;

    bool show_tree_recursive(Node* entity);
};
