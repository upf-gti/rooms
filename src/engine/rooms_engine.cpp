#include "rooms_engine.h"

#include "framework/entities/entity_mesh.h"

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

	if (error) return error;

	EntityMesh* torus = new EntityMesh();
	torus->get_mesh()->load_mesh("data/meshes/torus/torus.obj");
	torus->scale(glm::vec3(0.25));
	torus->translate(glm::vec3(-1.0f, 0.0, 0.0));

	EntityMesh* cube = new EntityMesh();
	cube->get_mesh()->load_mesh("data/meshes/cube/cube.obj");
	cube->scale(glm::vec3(0.25));
	cube->translate(glm::vec3(1.0f, 0.0, 0.0));

	entities.push_back(torus);
	entities.push_back(cube);

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

	Engine::render();
}
