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

	// Tooling

	tools[SCULPTING] = (EditorTool*) new SculptTool();
	tools[COLOR] = (EditorTool*) new ColoringTool();

	for (auto t : tools) {
		if(t) t->initialize();
	}

	tool_controller.set_workspace({ 88.f, 48.f }, XR_BUTTON_A, POSE_AIM);

	// Tools Menu
	{
		tool_controller.make_button("sculpt_mode", { 8.f, 8.f }, { 32.f, 32.f }, colors::GREEN);
		tool_controller.make_button("color_mode", { 48.f , 8.f }, { 32.f, 32.f }, colors::PURPLE);
	}

	// Events
	{
		tool_controller.connect("sculpt_mode", [&](const std::string& signal, float value) {
			enable_tool(SCULPTING);
		});
		tool_controller.connect("color_mode", [&](const std::string& signal, float value) {
			enable_tool(COLOR);
		});
	}
	enable_tool(SCULPTING);
	return error;
}

void RoomsEngine::update(float delta_time)
{
	Engine::update(delta_time);

	entities[0]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[1]->rotate(1.6f * delta_time, glm::vec3(0.0, 0.0, 1.0));
	entities[2]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));
	entities[3]->rotate(1.6f * delta_time, glm::vec3(1.0, 0.0, 0.0));

	if(current_tool != NONE)
		tools[current_tool]->update(delta_time);

	tool_controller.update(delta_time);

	if (Input::is_button_pressed(XR_BUTTON_B))
	{
		tool_controller.enabled = true;

		for (auto t : tools) {
			if(t) t->stop();
		}
	}
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

#ifdef XR_SUPPORT
	if (current_tool != NONE)
	{
		tools[current_tool]->render_scene();
		tools[current_tool]->render_ui();
	}
	tool_controller.render();
#endif

	Engine::render();
}

void RoomsEngine::enable_tool(eTool tool)
{
	tool_controller.enabled = false;

	if (current_tool != NONE)
	{
		tools[current_tool]->stop();
	}
	
	tools[tool]->start();
	current_tool = tool;
}