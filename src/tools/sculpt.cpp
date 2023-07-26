#include "sculpt.h"
#include "utils.h"
#include "framework/input.h"



inline bool is_tool_being_used() {
#ifdef XR_SUPPORT
	return Input::is_key_pressed(GLFW_KEY_A) || Input::get_trigger_value(HAND_RIGHT) > 0.5;
#else
	return Input::is_key_pressed(GLFW_KEY_A);
#endif
}


void SculptTool::initialize(Renderer *edit_render) {
	edit_renderer = (RaymarchingRenderer*)edit_render;

	// Config UI
	ui_controller.set_workspace({ 256.f, 140.f }, XR_BUTTON_A, POSE_AIM, HAND_LEFT, HAND_RIGHT);

	// UI Layout
	{
		ui_controller.make_button("smooth_toggle", { 16.f, 16.f }, { 32.f, 32.f }, colors::GREEN);

		ui_controller.make_button("mirror_toggle", { 84.f, 16.f }, { 32.f, 32.f }, colors::PURPLE);

		// RGB color selector
		ui_controller.make_slider("color_R_slide", { 16.0f, 42.0f }, { 128.f, 32.f }, colors::RED);
		ui_controller.make_slider("color_G_slide", { 16.0f, 84.0f }, { 128.f, 32.f }, colors::GREEN);
		ui_controller.make_slider("color_B_slide", { 16.0f, 126.0f }, { 128.f, 32.f }, colors::BLUE);
	}

	// UI events
	{
		ui_controller.connect("smooth_toggle", [edit_to_add  = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->operation = (sdOperation)((edit_to_add->operation >= 4) ? (edit_to_add->operation - 4) : (edit_to_add->operation + 4));
		});
		ui_controller.connect("mirror_toggle", [use_mirror = &use_mirror](const std::string& signal, float value) {
			use_mirror[0] = !use_mirror[0];
		});
		ui_controller.connect("color_R_slide", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->color.x = 1.0f - value;
		});
		ui_controller.connect("color_G_slide", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->color.y = 1.0f - value;
		});
		ui_controller.connect("color_B_slide", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->color.z = 1.0f - value;
		});
	}
}
void SculptTool::clean() {}

void SculptTool::update(float delta_tim) {

	ui_controller.update(delta_tim);

	if (is_tool_being_used()) {

#ifdef XR_SUPPORT
		edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);
#else
		edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
#endif
		// Store the end of the sausage on the unused size attribute
		if (edit_to_add.primitive == SD_CAPSULE && !is_sausage_start_setted) {
			edit_to_add.size = edit_to_add.position;
			is_sausage_start_setted = true;
			return;
		}

		edit_renderer->push_edit(edit_to_add);

		// If the mirror is activated, mirror using the plane, and add another edit to the list
		if (use_mirror) {
			float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
			edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

			// Also mirror the other side of the capsule
			if (edit_to_add.primitive == SD_CAPSULE) {
				dist_to_plane = glm::dot(mirror_normal, edit_to_add.size - mirror_origin);
				edit_to_add.size = edit_to_add.size - mirror_normal * dist_to_plane * 2.0f;
			}

			edit_renderer->push_edit(edit_to_add);
		}

		is_sausage_start_setted = false;
	}
}

void SculptTool::render_scene() {}

void SculptTool::render_ui() {
	ui_controller.render();
}