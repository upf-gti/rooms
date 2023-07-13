#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"


namespace ui {

	RaymarchingRenderer* renderer = nullptr;

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

	void Controller::render( RaymarchingRenderer* _renderer )
	{
		renderer = _renderer;

		// Render workspace

		uint8_t hand = workspace.hand;
		uint8_t pose = POSE_AIM;

		glm::vec3 position = Input::get_controller_position(hand, pose);
		glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);

		// Set above the controller
		// workspace_transform = glm::translate(workspace_transform, glm::vec3(0.f, 1.f, 0.f));

		sEdit edit{
			.position = position,
			.primitive = SD_BOX,
			.color = glm::vec3(1.f),
			.operation = OP_UNION,
			.size = glm::vec3(0.1f, 0.1f, 0.001f),
			.radius = 0.f
		};

		edit.position.y -= 1.f;

		if(Input::was_button_pressed(XR_BUTTON_X))
			renderer->push_edit(edit);

		// Mirror
		if (make_button({ 0, 0 }, { 128, 128 })) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}
	}

	bool Controller::make_button(glm::vec2 pos, glm::vec2 size, const char* texture)
	{
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

		// ...

		return pressed;
	}
}