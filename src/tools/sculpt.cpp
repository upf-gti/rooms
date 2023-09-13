#include "sculpt.h"
#include "utils.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

inline bool is_tool_being_used() {
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
	ui_controller.set_workspace({ ww, 64.f }, XR_BUTTON_A, POSE_AIM, HAND_LEFT, HAND_RIGHT);

	// UI Layout
	{
		ui_controller.make_submenu("modes", { 24.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
		ui_controller.make_button("normal", { 24.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/normal.png");
		ui_controller.make_button("stamp", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
		ui_controller.make_button("colors", { 80.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/colors.png");
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
		// ...

		// Primitives
		ui_controller.connect("sphere", [&](const std::string& signal, float value) { edit_to_add.primitive = SD_SPHERE; });
		ui_controller.connect("cube", [&](const std::string& signal, float value) { edit_to_add.primitive = SD_BOX; });

		// Tools
		ui_controller.connect("mirror", [&](const std::string& signal, float value) { use_mirror = !use_mirror; });
	}

	// UI Layout
	//{
	//	ui_controller.make_button("enable_smooth", { 8.f, 8.f }, { 32.f, 32.f }, colors::GREEN);

	//	ui_controller.make_button("toggle_union_substract", { 48.f, 8.f }, { 32.f, 32.f }, colors::PURPLE);

	//	// ui_controller.make_button("enable_sausage", { 130.f, 0.f }, { 50.f, 25.f }, colors::RED);

	//	ui_controller.make_slider("radius_slider", 0.1f, { 8.0f, 42.0f }, { 112.0f, 16.0f }, colors::YELLOW);

	//	ui_controller.make_color_picker("edit_color", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f }, { 130.0f, 8.0f }, { 64.f, 16.f });
	//}

	//// UI events
	//{
	//	ui_controller.connect("enable_smooth", [edit_to_add  = &edit_to_add](const std::string& signal, float value) {
	//		edit_to_add->operation = (sdOperation)((edit_to_add->operation >= 4) ? (edit_to_add->operation - 4) : (edit_to_add->operation + 4));
	//	});
	//	ui_controller.connect("toggle_union_substract", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
	//		switch (edit_to_add->operation)
	//		{
	//		case OP_UNION:
	//			edit_to_add->operation = OP_SUBSTRACTION;
	//			break;
	//		case OP_SUBSTRACTION:
	//			edit_to_add->operation = OP_UNION;
	//		case OP_SMOOTH_UNION:
	//			edit_to_add->operation = OP_SMOOTH_SUBSTRACTION;
	//			break;
	//		case OP_SMOOTH_SUBSTRACTION:
	//			edit_to_add->operation = OP_SMOOTH_UNION;
	//		default:
	//			break;
	//		}
	//	});
	//	
	//	ui_controller.connect("radius_slider", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
	//		edit_to_add->radius = (value / 10.0f * 0.5f) + 0.01f;
	//	});
	//	/*ui_controller.connect("on_edit_toggle", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
	//		edit_to_add->primitive = (edit_to_add->primitive == SD_CAPSULE) ? SD_SPHERE : SD_CAPSULE;
	//	});*/
	//	ui_controller.connect("edit_color", [edit_to_add = &edit_to_add](const std::string& signal, const Color& color) {
	//		edit_to_add->color = color;
	//	});
	//}
}

void SculptTool::clean()
{

}

void SculptTool::update(float delta_time)
{
	if (!enabled) return;

	ui_controller.update(delta_time);
	
	if (Input::is_button_pressed(XR_BUTTON_B))
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

	if (renderer->get_openxr_available()) {
		edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
		glm::quat rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT));
		edit_to_add.rotation = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
	} else {
		edit_to_add.position = glm::vec3(0.0f, 0.5f, 0.0f);
		edit_to_add.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	if (is_rotation_being_used()) {

		if (!rotation_started) {
			initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
		}

		glm::quat rotation_diff = glm::inverse(Input::get_controller_rotation(HAND_LEFT)) * glm::inverse(initial_hand_rotation);

		//sculpt_rotation = rotation_diff * sculpt_rotation;

		renderer->set_sculpt_rotation(glm::inverse(Input::get_controller_rotation(HAND_LEFT)));

		rotation_started = true;
	}
	else {
		rotation_started = false;
	}

	edit_to_add.position = glm::vec4(edit_to_add.position, 1.0f);

	if (!sculpt_started) {
		renderer->set_sculpt_start_position(edit_to_add.position);
	}

	renderer->set_preview_edit(edit_to_add);

	if (is_tool_being_used()) {

		if (!sculpt_started) {
			sculpt_started = true;
		}

		if (!renderer->get_openxr_available()) {
			edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
			glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
			edit_to_add.size = glm::vec3(0.01f, 0.01f, 0.05f);
			glm::quat rotation_random = glm::inverse(glm::quat(euler_angles));
			edit_to_add.rotation = glm::vec4(rotation_random.x, rotation_random.y, rotation_random.z, rotation_random.w);
		}

		// Store the end of the sausage on the unused size attribute
		//if (edit_to_add.primitive == SD_CAPSULE && !is_sausage_start_setted) {
		//	edit_to_add.size = edit_to_add.position;
		//	is_sausage_start_setted = true;
		//	has_trigger_used = true;
		//	return;
		//}

		renderer->push_edit(edit_to_add);

		// If the mirror is activated, mirror using the plane, and add another edit to the list
		if (use_mirror) {
			float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
			edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

			// Also mirror the other side of the capsule
			//if (edit_to_add.primitive == SD_CAPSULE) {
			//	dist_to_plane = glm::dot(mirror_normal, edit_to_add.size - mirror_origin);
			//	edit_to_add.size = edit_to_add.size - mirror_normal * dist_to_plane * 2.0f;
			//}

			renderer->push_edit(edit_to_add);
		}

		is_sausage_start_setted = false;
		has_trigger_used = true;
	} else {
		has_trigger_used = false;
	}
}

void SculptTool::render_scene()
{
	if (edit_to_add.operation == OP_SUBSTRACTION ||
		edit_to_add.operation == OP_SMOOTH_SUBSTRACTION)
	{
		mesh_preview->set_model(Input::get_controller_pose(ui_controller.get_workspace().select_hand));
		mesh_preview->scale(glm::vec3(edit_to_add.radius));
		mesh_preview->render();
	}
}

void SculptTool::render_ui()
{
	if (!enabled) return;
	
	ui_controller.render();
}