#include "sculpt.h"
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

inline bool is_rotation_being_used() {
	return Input::get_trigger_value(HAND_LEFT) > 0.5;
}

void SculptTool::initialize()
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
		ui_controller.make_submenu("modes", { 24.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
			ui_controller.make_button("normal", { 24.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/normal.png");
			ui_controller.make_button("stamp", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
			ui_controller.make_color_picker("colors", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f } , { 80.f, 36.f }, { 32.f, 8.f });
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
			edit_to_add.radius = edit_to_add.size.x;
		});
		ui_controller.connect("cube", [&](const std::string& signal, float value) { 
			edit_to_add.primitive = SD_BOX; 
			mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj")); 
			edit_to_add.size = glm::vec3(edit_to_add.radius);
			edit_to_add.radius = 0.f;
		});

		// Tools
		ui_controller.connect("mirror", [&](const std::string& signal, float value) { use_mirror = !use_mirror; });
	}
}

void SculptTool::clean()
{

}

void SculptTool::update(float delta_time)
{
	EditorTool::update(delta_time);

	if (!enabled) return;

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

	// Update edit position
	if (renderer->get_openxr_available()) {
		edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
		edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT));
	} else {
		edit_to_add.position = glm::vec3(0.0f, 0.5f, 0.0f);
		edit_to_add.rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	}

	// Update edit dimensions depending on primitive
	float size_multipler = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1;
	switch (edit_to_add.primitive)
	{
	case SD_SPHERE:
		edit_to_add.radius = glm::clamp(size_multipler + edit_to_add.radius, 0.001f, 0.1f);
		break;
	case SD_BOX:
		edit_to_add.size = glm::clamp(glm::vec3(size_multipler) + edit_to_add.size, glm::vec3(0.001f), glm::vec3(0.1f));
	default:
		break;
	}

	// Rotate the scene TODO: when ready move this out of tool to engine
	if (is_rotation_being_used()) {

		if (!rotation_started) {
			initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
			initial_hand_translation = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f);
		}

		rotation_diff = glm::inverse(initial_hand_rotation) * glm::inverse(Input::get_controller_rotation(HAND_LEFT));
		translation_diff = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f) - initial_hand_translation;

		renderer->set_sculpt_rotation(sculpt_rotation * rotation_diff);
		renderer->set_sculpt_start_position(sculpt_start_position + translation_diff);

		rotation_started = true;
	}
	else {
		if (rotation_started) {
			sculpt_rotation = sculpt_rotation * rotation_diff;
			sculpt_start_position = sculpt_start_position + translation_diff;
			rotation_started = false;
			rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
		}
	}

	edit_to_add.position = glm::vec4(edit_to_add.position, 1.0f);

	// Set center of sculpture
	if (!sculpt_started) {
		sculpt_start_position = edit_to_add.position;
		renderer->set_sculpt_start_position(sculpt_start_position);
	}

	// Set position of the preview edit
	renderer->set_preview_edit(edit_to_add);

	// Sculpting (adding edits)
	if (is_tool_activated()) {

		sculpt_started = true;

		// For debugging sculpture without a headset
		if (!renderer->get_openxr_available()) {
			edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
			glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
			edit_to_add.size = glm::vec3(0.01f, 0.01f, 0.05f);
			edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
		}

		use_tool();
	} 
}

void SculptTool::render_scene()
{
	// Render a hollowed Edit
	if (edit_to_add.operation == OP_SUBSTRACTION ||
		edit_to_add.operation == OP_SMOOTH_SUBSTRACTION)
	{
		mesh_preview->set_model(Input::get_controller_pose(ui_controller.get_workspace().select_hand));

		switch (edit_to_add.primitive)
		{
		case SD_SPHERE:
			mesh_preview->scale(glm::vec3(edit_to_add.radius));
			break;
		case SD_BOX:
			mesh_preview->scale(edit_to_add.size);
		default:
			break;
		}

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
	if (EditorTool::use_tool()) {
		renderer->push_edit(edit_to_add);

		// If the mirror is activated, mirror using the plane, and add another edit to the list
		if (use_mirror) {
			float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
			edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

			renderer->push_edit(edit_to_add);
		}
		
		return true;
	}

	return false;
}
