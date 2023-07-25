#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"
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

	// UI

	ui_controller.set_workspace({ 256.f, 64.f  }, XR_BUTTON_A, POSE_AIM, HAND_LEFT, HAND_RIGHT);

	for (int i = 0; i < 2; ++i)
	{																				  // base, hover, active colors
		ui_controller.make_button("on_button_a", { 16.f * (i + 1) + i * 32.f, 16.f }, { 32.f, 32.f }, colors::GREEN);
	}

	ui_controller.make_text("on_slider_changed", { 0.f, 0.f }, colors::RED, 25.f);

	ui_controller.make_slider("on_slider_changed", { 112.f, 16.f }, { 128.f, 32.f }, colors::PURPLE);

	ui_controller.connect("on_button_a", [](const std::string& signal, float value) {
		std::cout << "Signal: " << signal << std::endl;
	});

	ui_controller.connect("on_slider_changed", [torus](const std::string& signal, float value) {
		std::cout << "Signal: " << signal << ", Value: " << value << std::endl;
		torus->set_translation(glm::vec3(value, 0.f, 0.f));
		torus->scale(glm::vec3(0.25f));
	});

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));
	entities[2]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));

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
