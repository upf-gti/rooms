#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

EntityMesh* origin_el = nullptr;
EntityMesh* intersection_el = nullptr;

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

		intersection_el = new EntityMesh();
		intersection_el->get_mesh()->load("data/meshes/cube.obj");
	}

	void Controller::render()
	{
		uint8_t hand = workspace.hand;
		uint8_t pose = workspace.root_pose;

		// Render helpers

		glm::mat4x4 raycast_transform = Input::get_controller_pose(workspace.select_hand, pose);
		origin_el->set_model(raycast_transform);
		origin_el->rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
		origin_el->scale(glm::vec3(0.1f));

		// Render workspace

		glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);
		glm::vec3 workspace_position = workspace_transform[3];

		// Render quad with workspace size
		EntityMesh* ui_el = new EntityMesh();
		ui_el->destroy_after_render = true;
		ui_el->get_mesh()->create_quad(workspace.size.x, workspace.size.y, hovered ? GREEN : glm::vec3(1,1,1));
		//ui_el->set_model( glm::translate(glm::mat4x4(1.f), workspace_position) );
		ui_el->set_model(workspace_transform);
		ui_el->rotate(glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));

		global_transform = ui_el->get_global_matrix();

		ui_el->render();

		// Render buttons

		// TEST WORKSPACE
		// make_button({ 0.f, 0.f }, workspace.size, hovered ? BLUE : GREEN);

		/*if (make_button({ 0.125f, 0.05f }, { 0.125f, 0.05f }, pressed ? RED : PURPLE)) {

			std::cout << "Button pressed!" << std::endl;

			pressed = !pressed;
		}

		if (make_button({ 0.f, 0.05f }, { 0.125f, 0.05f }, pressed ? GREEN : BLUE)) {

			std::cout << "Button pressed!" << std::endl;

			pressed = !pressed;
		}*/

		origin_el->render(); 
	}

	void Controller::update(float delta_time)
	{
		// ...
	}

	bool Controller::make_button(glm::vec2 pos, glm::vec2 size, glm::vec3 color, const char* texture)
	{
		
		/*
		*	To workspace local size
		*/
		
		pos -= (workspace.size - size - pos);

		/*
		*	Render button
		*/

		// Render quad in local workspace position
		EntityMesh* ui_el = new EntityMesh();
		ui_el->destroy_after_render = true;
		ui_el->get_mesh()->create_quad(size.x, size.y, color);
		ui_el->set_model(global_transform);
		ui_el->translate(glm::vec3(pos.x, pos.y, -1e-3f));
		// ui_el->render();

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

		// Position
		glm::vec3 quad_position = Input::get_controller_position(hand, pose);// +glm::vec3(pos.x, -1e-3f, pos.y);

		// Rotation
		glm::mat4x4 hand_pose = Input::get_controller_pose(hand, pose);
		// glm::quat quad_rotation = glm::quat(0.707f, -0.707f, 0.f, 0.f);
		// glm::quat quad_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
		// glm::quat quad_rotation = Input::get_controller_rotation(hand, pose);
		glm::quat quad_rotation = glm::quat_cast(glm::rotate(hand_pose, glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f)));
		
		glm::vec3 intersection;
		float collision_dist = -1.f;

		if (Input::was_button_pressed(workspace.select_button))
		{
			std::cout << "" << std::endl;
		}

		// Check hover (intersects)
		hovered = intersection::ray_quad(
						ray_origin,
						ray_direction,
						quad_position,
						size,
						quad_rotation,
						intersection,
						collision_dist
		);

		if (collision_dist > 0)
		{
			intersection_el->set_model(global_transform);
			intersection_el->set_model(glm::translate(glm::mat4x4(1.f), intersection));
			intersection_el->scale(glm::vec3(0.02f));
			intersection_el->render();
		}

		// Check pressed (hover and button pressed)
		bool pressed = hovered && Input::was_button_pressed( workspace.select_button );

		return pressed;
	}
}