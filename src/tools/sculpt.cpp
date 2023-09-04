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

void SculptTool::initialize()
{
	renderer = dynamic_cast<RaymarchingRenderer*>(Renderer::instance);

	// Config UI
	ui_controller.set_workspace({ 190.f, 140.f }, XR_BUTTON_A, POSE_AIM, HAND_LEFT, HAND_RIGHT);

	// UI Layout
	{
		ui_controller.make_button("on_smooth_toggle", { 10.f, 0.f }, { 50.f, 25.f }, colors::GREEN);

		ui_controller.make_button("on_mirror_toggle", { 70.f, 0.f }, { 50.f, 25.f }, colors::PURPLE);

		ui_controller.make_button("on_edit_toggle", { 130.f, 0.f }, { 50.f, 25.f }, colors::RED);

		ui_controller.make_slider("on_radius_slider", 0.1f, { 16.0f, 34.0f }, { 128.0f, 32.0f }, colors::YELLOW);

		ui_controller.make_color_picker("on_color_pick", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f }, { 16.0f, 82.0f }, { 64.f, 16.f });
	}

	// UI events
	{
		ui_controller.connect("on_smooth_toggle", [edit_to_add  = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->operation = (sdOperation)((edit_to_add->operation >= 4) ? (edit_to_add->operation - 4) : (edit_to_add->operation + 4));
		});
		ui_controller.connect("on_mirror_toggle", [use_mirror = &use_mirror](const std::string& signal, float value) {
			use_mirror[0] = !use_mirror[0];
		});
		ui_controller.connect("on_radius_slider", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->radius = (value / 10.0f * 0.5f) + 0.01f;
		});
		ui_controller.connect("on_edit_toggle", [edit_to_add = &edit_to_add](const std::string& signal, float value) {
			edit_to_add->primitive = (edit_to_add->primitive == SD_CAPSULE) ? SD_SPHERE : SD_CAPSULE;
		});
		ui_controller.connect("on_color_pick", [edit_to_add = &edit_to_add](const std::string& signal, const Color& color) {
			edit_to_add->color = color;
		});
	}
}

void SculptTool::clean()
{

}

void SculptTool::update(float delta_time)
{
	ui_controller.update(delta_time);
	
	edit_to_add.position = Input::get_controller_position(HAND_RIGHT) - glm::vec3(0.0f, 1.0f, 0.0f);

	if (is_tool_being_used()) {

#ifdef XR_SUPPORT
		float curr_trigger_value = Input::get_trigger_value(HAND_RIGHT);

		if (edit_to_add.primitive == SD_CAPSULE && has_trigger_used) {
			return;
		}

#else
		edit_to_add.position = glm::vec3(0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1), 0.4 * (random_f() * 2 - 1));
#endif
		// Store the end of the sausage on the unused size attribute
		if (edit_to_add.primitive == SD_CAPSULE && !is_sausage_start_setted) {
			edit_to_add.size = edit_to_add.position;
			is_sausage_start_setted = true;
			has_trigger_used = true;
			return;
		}

		renderer->push_edit(edit_to_add);

		// If the mirror is activated, mirror using the plane, and add another edit to the list
		if (use_mirror) {
			float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
			edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

			// Also mirror the other side of the capsule
			if (edit_to_add.primitive == SD_CAPSULE) {
				dist_to_plane = glm::dot(mirror_normal, edit_to_add.size - mirror_origin);
				edit_to_add.size = edit_to_add.size - mirror_normal * dist_to_plane * 2.0f;
			}

			renderer->push_edit(edit_to_add);
		}

		is_sausage_start_setted = false;
		has_trigger_used = true;
	} else {
		has_trigger_used = false;

		renderer->set_preview_edit(edit_to_add);
	}
}

void SculptTool::render_scene()
{

}

void SculptTool::render_ui()
{
	ui_controller.render();
}