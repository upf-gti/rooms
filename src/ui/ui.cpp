#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"

EntityMesh* origin_el = nullptr;
EntityMesh* debug_el = nullptr;

namespace ui {

	void Controller::set_workspace(glm::vec2 _workspace_size, uint8_t _select_button, uint8_t _root_pose, uint8_t _hand, uint8_t _select_hand)
	{
		global_scale = 0.001f;

		workspace = {
			.size = _workspace_size * global_scale,
			.select_button = _select_button,
			.root_pose = _root_pose,
			.hand = _hand,
			.select_hand = _select_hand
		};

		origin_el = new EntityMesh();
		origin_el->get_mesh()->load("data/meshes/raycast.obj");

		debug_el = new EntityMesh();
		debug_el->get_mesh()->load("data/meshes/cube/cube.obj");
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

		// Render buttons (example)

		for (int i = 0; i < 2; ++i)
		{																				  // base, hover, active colors
			make_button("on_button_a", { 16.f * (i + 1) + i * 32.f, 16.f }, { 32.f, 32.f }, { colors::GREEN, colors::PURPLE, colors::RED });
		}

		make_slider("on_slider_changed", { 112.f, 16.f }, { 128.f, 32.f }, { colors::GREEN, colors::PURPLE, colors::YELLOW });
	}

	void Controller::update(float delta_time)
	{
		// ...
	}

	void Controller::make_button(const std::string& signal, glm::vec2 pos, glm::vec2 size, const ButtonColorData& data)
	{
		pos *= global_scale;
		size *= global_scale;

		// Clamp to workspace limits
		size = glm::clamp(size, glm::vec2(0.f), workspace.size - pos);

		// To workspace local size
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
		bool hovered = intersection::ray_quad(
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

		if(was_pressed)
			emit_signal(signal);
	}

	void Controller::make_slider(const std::string& signal, glm::vec2 pos, glm::vec2 size, const ButtonColorData& data)
	{
		pos *= global_scale;
		size *= global_scale;

		// Clamp to workspace limits
		size = glm::clamp(size, glm::vec2(0.f), workspace.size - pos);

		// Thumb data
		glm::vec2 thumb_size = { size.y, size.y };
		float thumb_pos = current_slider_pos * global_scale - workspace.size.x + thumb_size.x + pos.x * 2.f;

		// To workspace local size
		pos -= (workspace.size - size - pos);

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_slider = new EntityMesh();
		e_slider->destroy_after_render = true;
		e_slider->set_model(global_transform);
		e_slider->translate(glm::vec3(pos.x, pos.y, -1e-3f));
		e_slider->get_mesh()->create_quad(size.x, size.y, colors::RED);
		e_slider->render();
		
		EntityMesh* e_thumb = new EntityMesh();
		e_thumb->destroy_after_render = true;
		e_thumb->set_model(global_transform);
		e_thumb->translate(glm::vec3(thumb_pos, pos.y, -2e-3f));

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
		glm::vec3 quad_position = e_thumb->get_translation();
		glm::vec3 slider_quad_position = e_slider->get_translation();
		glm::quat quad_rotation = glm::quat_cast(global_transform);

		// Check hover with thumb
		glm::vec3 intersection;
		float collision_dist;
		bool thumb_hovered = intersection::ray_quad(
			ray_origin,
			ray_direction,
			quad_position,
			thumb_size,
			quad_rotation,
			intersection,
			collision_dist
		);

		// Check hover with slider background to move thumb
		bool slider_hovered = intersection::ray_quad(
			ray_origin,
			ray_direction,
			slider_quad_position,
			size,
			quad_rotation,
			intersection,
			collision_dist
		);

		/*
		*	Create mesh and render thumb
		*/

		bool is_pressed = thumb_hovered && Input::is_button_pressed(workspace.select_button);
		bool was_pressed = thumb_hovered && Input::was_button_pressed(workspace.select_button);
		e_thumb->get_mesh()->create_quad(thumb_size.x, thumb_size.y, is_pressed ? data.active_color : (thumb_hovered ? data.hover_color : data.base_color));
		e_thumb->render();

		if (is_pressed)
		{
			max_slider_pos = (size.x * 2.f - thumb_size.x * 2.f) / global_scale;
			current_slider_pos = glm::clamp((intersection.x + size.x - thumb_size.x) / global_scale, 0.f, max_slider_pos);
			emit_signal(signal, glm::clamp(current_slider_pos / max_slider_pos, 0.f, 1.f));
		}
	}

	void Controller::connect(const std::string& name, std::function<void(const std::string&, float)> callback)
	{
		auto it = signals.find(name);
		if (it != signals.end())
			return;

		signals[name] = callback;
	}

	bool Controller::emit_signal(const std::string& name, float value)
	{
		auto it = signals.find(name);
		if (it == signals.end())
			return false;

		signals[name](name, value);
		return true;
	}
}