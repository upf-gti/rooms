#include "rooms_engine.h"

#include "framework/entities/entity_mesh.h"

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

	if (error) return error;

	EntityMesh* entity = new EntityMesh();
	entity->get_mesh()->load_mesh("data/meshes/torus.obj");

	entities.push_back(entity);

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

	Engine::render();
}
