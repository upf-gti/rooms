#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "ui/ui.h"

ui::Controller ui_controller;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

	if (error) return error;

	EntityMesh* torus = new EntityMesh();
	torus->get_mesh()->load("data/meshes/torus.obj");
	torus->scale(glm::vec3(0.25));
	torus->translate(glm::vec3(1.0, 0.0, 0.0));

	EntityMesh* cube = new EntityMesh();
	cube->get_mesh()->load("data/meshes/cube.obj");
	cube->scale(glm::vec3(0.25));
	cube->translate(glm::vec3(-1.0, 0.0, 0.0));

	entities.push_back(torus);
	entities.push_back(cube);

	// UI

	ui_controller.set_workspace({ 0.5f, 0.2f });

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));

	ui_controller.update(delta_time);
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

	ui_controller.render();

	Engine::render();
}
