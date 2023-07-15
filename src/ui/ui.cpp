#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

EntityMesh* origin_el = nullptr;
EntityMesh* debug_el = nullptr;

bool hovered = false;

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

		origin_el = new EntityMesh();
		origin_el->get_mesh()->load("data/meshes/raycast.obj");

		debug_el = new EntityMesh();
		debug_el->get_mesh()->load("data/meshes/cube.obj");
	}

	void Controller::render()
	{
		uint8_t hand = workspace.hand;
		uint8_t pose = workspace.root_pose;

		// Render raycast helper

		glm::mat4x4 raycast_transform = Input::get_controller_pose(workspace.select_hand, pose);
		origin_el->set_model(raycast_transform);
		origin_el->rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
		origin_el->scale(glm::vec3(0.1f));
		origin_el->render();

		if (Input::get_grab_value(hand) < 0.5f)
			return;

		// Render workspace

		glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);
		glm::vec3 workspace_position = workspace_transform[3];

		// Render quad with workspace size
		EntityMesh* ui_el = new EntityMesh();
		ui_el->destroy_after_render = true;
		ui_el->get_mesh()->create_quad(workspace.size.x, workspace.size.y);
		ui_el->set_model(workspace_transform);
		ui_el->rotate(glm::radians(120.f), glm::vec3(1.f, 0.f, 0.f));
		ui_el->render();

		global_transform = ui_el->get_global_matrix();

		// Render buttons

		// TEST FULL WORKSPACE AS BUTTON
		// make_button({ 0.f, 0.f }, workspace.size, hovered ? BLUE : GREEN);

		if (make_button({ 0.125f, 0.05f }, {0.125f, 0.05f}, {
				.base_color = colors::GREEN,
				.hover_color = colors::PURPLE,
				.active_color = colors::BLUE
			})) {

			std::cout << "Button pressed!" << std::endl;
		}

		if (make_button({ 0.025f, 0.025f }, { 0.05f, 0.05f }, {
				.base_color = colors::YELLOW,
				.hover_color = colors::PURPLE,
				.active_color = colors::BLUE
			})) {

			std::cout << "Button pressed!" << std::endl;
		}
	}

	void Controller::update(float delta_time)
	{
		// ...
	}

	bool Controller::make_button(glm::vec2 pos, glm::vec2 size, const ButtonColorData& data)
	{
		/*
		*	To workspace local size
		*/
		
		pos -= (workspace.size - size - pos);

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_button = new EntityMesh();
		e_button->destroy_after_render = true;
		e_button->set_model(global_transform);
		e_button->translate(glm::vec3(pos.x, pos.y, -1e-3f));

		/*
		*	Manage intersection
		*/

		uint8_t hand = workspace.hand;
		uint8_t select_hand = workspace.select_hand;
		uint8_t pose = workspace.root_pose;

		// Ray
		glm::vec3 ray_origin = Input::get_controller_position(select_hand, pose);
		glm::mat4x4 select_hand_pose = Input::get_controller_pose(select_hand, pose);
		glm::vec3 ray_direction = get_front(select_hand_pose);

		// Quad
		glm::vec3 quad_position = e_button->get_translation();
		glm::quat quad_rotation = glm::quat_cast(global_transform);
		
		// Check hover (intersects)
		glm::vec3 intersection;
		float collision_dist;
		hovered = intersection::ray_quad(
						ray_origin,
						ray_direction,
						quad_position,
						size,
						quad_rotation,
						intersection,
						collision_dist
		);

		/*
		*	Create mesh and render button
		*/

		bool is_pressed = hovered && Input::is_button_pressed(workspace.select_button);
		bool was_pressed = hovered && Input::was_button_pressed(workspace.select_button);
		e_button->get_mesh()->create_quad(size.x, size.y, is_pressed ? data.active_color : (hovered ? data.hover_color : data.base_color));
		e_button->render();

		return was_pressed;
	}
}