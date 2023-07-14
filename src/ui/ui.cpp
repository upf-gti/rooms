#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

namespace ui {

	void Controller::set_workspace(glm::vec2 _workspace_size, uint8_t _select_button, uint8_t _root_pose, uint8_t _hand, uint8_t _select_hand)
	{
		workspace = {
			.size = _workspace_size,
			.select_button = _select_button,
			.root_pose = _root_pose,
			.hand = _hand,
			.select_hand = _select_hand
		};
	}

	void Controller::render()
	{
		// Render workspace

		uint8_t hand = workspace.hand;
		uint8_t pose = POSE_AIM;

		// glm::vec3 position = Input::get_controller_position(hand, pose);
		glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);

		// Render quad with workspace size
		EntityMesh* ui_el = new EntityMesh();
		ui_el->destroy_after_render = true;
		ui_el->get_mesh()->create_quad(workspace.size.x, workspace.size.y);
		ui_el->set_model(workspace_transform);
		ui_el->rotate(glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
		ui_el->translate(glm::vec3(0.f, 0.5f, 0.f));

		global_transform = ui_el->get_global_matrix();

		ui_el->render();

		// Mirror
		if (make_button({ 0.f, 0.f }, { 0.25f, 0.1f }, RED)) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}

		if (make_button({ 0.25f, 0.1f }, { 0.25f, 0.1f }, GREEN)) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}

		if (make_button({ 0.25f, 0.f }, { 0.25f, 0.1f }, BLUE)) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}

		if (make_button({ 0.f, 0.1f }, { 0.125f, 0.05f }, PURPLE)) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}
	}

	void Controller::update(float delta_time)
	{
		// ...
	}

	bool Controller::make_button(glm::vec2 pos, glm::vec2 size, glm::vec3 color, const char* texture)
	{
		// To worksize local

		pos = 2.f * pos + size;
		pos.x = -workspace.size.x + pos.x;
		pos.y = workspace.size.y - pos.y;

		/*
		*	Manage intersection
		*/

		uint8_t hand = workspace.hand;
		uint8_t select_hand = workspace.select_hand;
		uint8_t pose = workspace.root_pose;

		// Ray
		glm::vec3 ray_origin = Input::get_controller_position(select_hand, pose);
		glm::vec3 ray_direction = get_front(Input::get_controller_pose(select_hand, pose));

		// Position
		glm::vec3 quad_position = glm::vec3(pos.x, 0, pos.y);
		glm::vec3 workspace_position = Input::get_controller_position(hand, pose);

		quad_position += workspace_position;

		// glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);

		// Rotation
		glm::quat quad_rotation = Input::get_controller_rotation(hand, pose);
		float collision_dist;

		// Check hover (intersects)
		bool hovered = intersection::ray_quad(
						ray_origin,
						ray_direction,
						quad_position,
						size,
						quad_rotation,
						collision_dist
		);

		// Check pressed (hover and button pressed)
		bool pressed = hovered && Input::is_button_pressed( workspace.select_button );

		/*
		*	Render button
		*/

		// Render quad in local workspace position
		EntityMesh* ui_el = new EntityMesh();
		ui_el->destroy_after_render = true;
		ui_el->get_mesh()->create_quad(size.x, size.y, color);
		ui_el->set_model(global_transform);
		ui_el->translate(glm::vec3(pos.x, pos.y, -1e-3f));
		ui_el->render();

		return pressed;
	}
}