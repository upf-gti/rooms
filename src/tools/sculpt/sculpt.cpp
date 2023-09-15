#include "sculpt.h"
#include "utils.h"
#include "framework/input.h"
#include "framework/entities/entity_mesh.h"
#include "graphics/mesh.h"
#include "graphics/shader.h"

void SculptTool::initialize()
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
		ui_controller.make_submenu("modes", { 24.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
		    ui_controller.make_button("normal", { 24.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/normal.png");
		    ui_controller.make_button("stamp", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
            ui_controller.make_submenu("colorize", { 80.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
                ui_controller.make_color_picker("colors", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f } , { 48.f, 68.f }, { 32.f, 8.f });
            ui_controller.close_submenu();
		ui_controller.close_submenu();
		
		ui_controller.make_submenu("primitives", { 52.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
		ui_controller.make_button("sphere", { 38.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
		ui_controller.make_button("cube", { 66.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/cube.png");
		ui_controller.close_submenu();

		ui_controller.make_submenu("tools", { 80.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/tools.png");
		ui_controller.make_button("mirror", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/mirror.png");
		ui_controller.close_submenu();
	}

	{
		// Modes
		ui_controller.connect("colors", [edit_to_add = &edit_to_add](const std::string& signal, const Color& color) {
			edit_to_add->color = color;
		});

		// Primitives
		ui_controller.connect("sphere", [&](const std::string& signal, float value) { 
			edit_to_add.primitive = SD_SPHERE; 
			mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
		});
		ui_controller.connect("cube", [&](const std::string& signal, float value) { 
			edit_to_add.primitive = SD_BOX; 
			mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj")); 
		});

		// Tools
		ui_controller.connect("mirror", [&](const std::string& signal, float value) {/* use_mirror = !use_mirror;*/ });
	}
}

void SculptTool::clean()
{

}

bool SculptTool::update(float delta_time)
{
	Tool::update(delta_time);

	if (!enabled) return false;

	ui_controller.update(delta_time);
	
	// Tool Operation changer
	if (Input::was_button_pressed(XR_BUTTON_Y))
	{
		switch (edit_to_add.operation)
		{
		case OP_UNION:
			edit_to_add.operation = OP_SUBSTRACTION;
			break;
		case OP_SUBSTRACTION:
			edit_to_add.operation = OP_UNION;
		case OP_SMOOTH_UNION:
			edit_to_add.operation = OP_SMOOTH_SUBSTRACTION;
			break;
		case OP_SMOOTH_SUBSTRACTION:
			edit_to_add.operation = OP_SMOOTH_UNION;
		default:
			break;
		}
	}

	// Sculpting (adding edits)
	if (is_tool_activated()) {

		// For debugging sculpture without a headset
		//if (!renderer->get_openxr_available()) {
		//	edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
		//	glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
		//	edit_to_add.dimensions = glm::vec4(0.01f, 0.01f, 0.05f, 0.01f);
		//	edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
		//}

        return use_tool();
	}

    return false;
}

void SculptTool::render_scene()
{
	// Render a hollowed Edit
	if (edit_to_add.operation == OP_SUBSTRACTION ||
		edit_to_add.operation == OP_SMOOTH_SUBSTRACTION)
	{
		mesh_preview->set_model(Input::get_controller_pose(ui_controller.get_workspace().select_hand));
		mesh_preview->scale(edit_to_add.dimensions);
		mesh_preview->render();
	}
}

void SculptTool::render_ui()
{
	if (!enabled) return;
	
	ui_controller.render();
}

bool SculptTool::use_tool()
{
	if (Tool::use_tool()) {
		return true;
	}

	return false;
}
