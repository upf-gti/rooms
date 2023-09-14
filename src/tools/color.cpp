#include "color.h"
#include "utils.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

inline bool is_tool_activated() {
#ifdef XR_SUPPORT
	return Input::is_key_pressed(GLFW_KEY_A) || Input::get_trigger_value(HAND_RIGHT) > 0.5;
#else
	return Input::is_key_pressed(GLFW_KEY_A);
#endif
}

void ColoringTool::initialize()
{
	renderer = dynamic_cast<RaymarchingRenderer*>(Renderer::instance);

	mesh_preview = new EntityMesh();
	mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
	mesh_preview->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
	mesh_preview->set_color(colors::WHITE);

	// Config UI
	const float ww = 128.f;
	ui_controller.set_workspace({ ww, ww }, XR_BUTTON_A, POSE_AIM, HAND_LEFT, HAND_RIGHT);

	// UI Layout
	{
		ui_controller.make_submenu("colorize", { 38.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/colors.png");
		ui_controller.make_color_picker("colors", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f }, { 38.f, 36.f }, { 64.f, 16.f });
		ui_controller.close_submenu();

		ui_controller.make_submenu("primitives", { 66.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
		ui_controller.make_button("sphere", { 38.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
		ui_controller.make_button("cube", { 66.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/cube.png");
		ui_controller.close_submenu();
	}

	{
		ui_controller.connect("colors", [edit_to_add = &edit_to_add](const std::string& signal, const Color& color) {
			edit_to_add->color = color;
		});

		ui_controller.connect("sphere", [&](const std::string& signal, float value) {
			edit_to_add.primitive = SD_SPHERE;
			mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
		});

		ui_controller.connect("cube", [&](const std::string& signal, float value) {
			edit_to_add.primitive = SD_BOX;
			mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj"));
		});
	}
}

void ColoringTool::clean()
{

}

void ColoringTool::update(float delta_time)
{
	EditorTool::update(delta_time);

	if (!enabled) return;

	ui_controller.update(delta_time);
	
#ifdef XR_SUPPORT
	edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
#else
	edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
#endif

	// Update common edit dimensions
	float size_multipler = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 0.1f;
	glm::vec3 new_dimensions = glm::clamp(size_multipler + glm::vec3(edit_to_add.dimensions), 0.001f, 0.1f);
	edit_to_add.dimensions = glm::vec4(new_dimensions, edit_to_add.dimensions.w);

	// Update primitive specific size
	size_multipler = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1f;
	edit_to_add.dimensions.w = glm::clamp(size_multipler + edit_to_add.dimensions.w, 0.001f, 0.1f);

	if (is_tool_activated())
	{
		use_tool();
	}
	else
	{
		renderer->set_preview_edit(edit_to_add);
	}
}

void ColoringTool::render_scene()
{
	mesh_preview->set_model(Input::get_controller_pose(ui_controller.get_workspace().select_hand));
	mesh_preview->scale(glm::vec3(edit_to_add.dimensions));
	mesh_preview->render();
}

void ColoringTool::render_ui()
{
	if (!enabled) return;

	ui_controller.render();
}

bool ColoringTool::use_tool()
{
	if (EditorTool::use_tool()) {
		renderer->push_edit(edit_to_add);
		return true;
	}

	return false;
}
