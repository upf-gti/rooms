#include "paint.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"

void PaintTool::initialize()
{
    Tool::initialize();

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

void PaintTool::clean()
{

}

bool PaintTool::update(float delta_time)
{
	Tool::update(delta_time);

	if (!enabled) return false;

	ui_controller.update(delta_time);
	
	edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);

	if (is_tool_activated())
	{
        return use_tool();
	}

    return false;
}

void PaintTool::render_scene()
{
	mesh_preview->set_model(Input::get_controller_pose(ui_controller.get_workspace().select_hand));
	mesh_preview->scale(glm::vec3(edit_to_add.dimensions));
	mesh_preview->render();
}

void PaintTool::render_ui()
{
	if (!enabled) return;

	ui_controller.render();
}

bool PaintTool::use_tool()
{
	if (Tool::use_tool()) {
		return true;
	}

	return false;
}
