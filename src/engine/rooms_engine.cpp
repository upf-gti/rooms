#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"

#include "tools/sculpt.h"
#include "tools/color.h"


int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_mirror_screen);

	if (error) return error;

	EntityMesh* torus = new EntityMesh();
	torus->set_mesh(Mesh::get("data/meshes/torus/torus.obj"));
	torus->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
	torus->scale(glm::vec3(0.25));
	torus->translate(glm::vec3(-1.0f, 0.0, 0.0));
	entities.push_back(torus);

	EntityMesh* cube = new EntityMesh();
	cube->set_mesh(Mesh::get("data/meshes/cube/cube.obj"));
	cube->set_shader(Shader::get("data/shaders/mesh_texture.wgsl"));
	cube->scale(glm::vec3(0.25));
	cube->translate(glm::vec3(1.0f, 0.0, 0.0));
	entities.push_back(cube);

	EntityMesh* cube2 = new EntityMesh();
	cube2->set_mesh(Mesh::get("data/meshes/cube/cube.obj"));
	cube2->set_shader(Shader::get("data/shaders/mesh_texture.wgsl"));
	cube2->scale(glm::vec3(0.25));
	cube2->translate(glm::vec3(4.0f, 0.0, 0.0));
	entities.push_back(cube2);

	TextEntity* text = new TextEntity("oppenheimer vs barbie");
	text->set_shader(Shader::get("data/shaders/sdf_fonts.wgsl"));
	text->set_color(colors::GREEN);
	text->set_scale(0.25f)->generate_mesh();
	text->translate(glm::vec3(0.0f, 1.0, 0.0));
	entities.push_back(text);

	right_controller_mesh = new EntityMesh();
	right_controller_mesh->set_mesh(Mesh::get("data/meshes/sphere.obj"));
	right_controller_mesh->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
	right_controller_mesh->scale(glm::vec3(0.25f));
	right_controller_mesh->translate(glm::vec3(1.0f, 1.0, 0.0));
	entities.push_back(right_controller_mesh);

	left_controller_mesh = new EntityMesh();
	left_controller_mesh->set_mesh(Mesh::get("data/meshes/sphere.obj"));
	left_controller_mesh->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
	left_controller_mesh->scale(glm::vec3(0.25f));
	left_controller_mesh->translate(glm::vec3(1.0f, 1.0, 0.0));
	entities.push_back(left_controller_mesh);

	// Tooling

	SculptTool *sculpting_tool =  new SculptTool();
	sculpting_tool->initialize();

	tools[SCULPTING] = (EditorTool*) sculpting_tool;


	ColoringTool* coloring_tool = new ColoringTool();
	coloring_tool->initialize();

	tools[COLOR] = (EditorTool*)coloring_tool;


	tool_selection_ui_controller.set_workspace({ 190.f, 140.f }, XR_BUTTON_A, POSE_AIM, HAND_RIGHT, HAND_LEFT);
	// Config UI
	{
		tool_selection_ui_controller.make_button("on_sculpt_button", { 10.f, 0.f }, { 50.f, 25.f }, colors::GREEN);

		tool_selection_ui_controller.make_button("on_color_button", { 70.f, 0.f }, { 50.f, 25.f }, colors::PURPLE);
	}
	// UI events
	{
		tool_selection_ui_controller.connect("on_sculpt_button", [current_tool = &current_tool](const std::string& signal, float value) {
			*current_tool = SCULPTING;
		});
		tool_selection_ui_controller.connect("on_color_button", [current_tool = &current_tool](const std::string& signal, float value) {
			*current_tool = COLOR;
		});
	}

	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));
	entities[2]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[3]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));

	right_controller_mesh->set_translation(Input::get_controller_position(HAND_RIGHT));
	right_controller_mesh->scale(glm::vec3(0.10f));
	left_controller_mesh->set_translation(Input::get_controller_position(HAND_LEFT));
	left_controller_mesh->scale(glm::vec3(0.10f));

	tools[current_tool]->update(delta_time);

	tool_selection_ui_controller.update(delta_time);
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

#ifdef XR_SUPPORT
	tools[current_tool]->render_ui();
	tool_selection_ui_controller.render();
#endif

	Engine::render();
}
