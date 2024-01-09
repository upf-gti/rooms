#pragma once

#include "engine.h"
#include "tools/sculpt/sculpt_editor.h"

class RoomsEngine : public Engine {

    static std::vector<Entity*> entities;
    static EntityMesh* skybox;
    static bool rotate_scene;

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

#ifdef __EMSCRIPTEN__
    static void set_skybox_texture(const std::string& filename);
    static void load_glb(const std::string& filename);
    static void toggle_rotation();
#endif
};
