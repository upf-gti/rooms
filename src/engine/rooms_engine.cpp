#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

    sculpt_editor.initialize();

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

    sculpt_editor.update(delta_time);
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

    sculpt_editor.render();

	Engine::render();
}
