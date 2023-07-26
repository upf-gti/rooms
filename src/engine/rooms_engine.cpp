#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"

#include "tools/sculpt.h"
#include "ui/ui_controller.h"

ui::Controller ui_controller;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

	if (error) return error;

	EntityMesh* torus = new EntityMesh();
	torus->set_mesh(Mesh::get("data/meshes/torus/torus.obj"));
	torus->scale(glm::vec3(0.25));
	torus->translate(glm::vec3(-1.0f, 0.0, 0.0));
	entities.push_back(torus);

	EntityMesh* cube = new EntityMesh();
	cube->set_mesh(Mesh::get("data/meshes/cube/cube.obj"));
	cube->scale(glm::vec3(0.25));
	cube->translate(glm::vec3(1.0f, 0.0, 0.0));
	entities.push_back(cube);

	TextEntity* text = new TextEntity("oppenheimer vs barbie");
	text->set_color(colors::GREEN)->set_scale(0.25f)->generate_mesh();
	text->translate(glm::vec3(0.0f, 1.0, -3.0));
	entities.push_back(text);

	// Tooling

	SculptTool *sculpting_tool =  new SculptTool();
	sculpting_tool->initialize();

	tools[SCULPTING] = (EditorTool*) sculpting_tool;

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));
	entities[2]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));

	//ui_controller.update(delta_time);

	tools[current_tool]->update(delta_time);
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

	//ui_controller.render();
#ifdef XR_SUPPORT
	tools[current_tool]->render_ui();
#endif

	Engine::render();
}
