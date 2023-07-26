#include "ui/ui_controller.h"
#include "ui/ui_widgets.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"
#include "framework/entities/entity_text.h"

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

		raycast_pointer = new EntityMesh();
		raycast_pointer->set_mesh(Mesh::get("data/meshes/raycast.obj"));

		workspace_element = new EntityMesh();
		Mesh* mesh = new Mesh();
		mesh->create_quad(workspace.size.x, workspace.size.y);
		workspace_element->set_mesh(mesh);
	}

	bool Controller::is_active()
	{
		return Input::get_grab_value(workspace.hand) > 0.5f;
	}

	void Controller::render()
	{
		if (!is_active()) return;

		workspace_element->render();

		for (auto widget : root) {
			widget->render();
		}

		raycast_pointer->render();
	}

	void Controller::update(float delta_time)
	{
		if (!is_active()) return;

		uint8_t hand = workspace.hand;
		uint8_t pose = workspace.root_pose;

		// Update raycast helper

		glm::mat4x4 raycast_transform = Input::get_controller_pose(workspace.select_hand, pose);
		raycast_pointer->set_model(raycast_transform);
		raycast_pointer->rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
		raycast_pointer->scale(glm::vec3(0.1f));

		// Update workspace

		glm::mat4x4 workspace_transform = Input::get_controller_pose(hand, pose);
		workspace_element->set_model(workspace_transform);
		workspace_element->rotate(glm::radians(120.f), glm::vec3(1.f, 0.f, 0.f));

		// Store global matrix for the rest of elements

		global_transform = workspace_element->get_global_matrix();

		// Update widgets using this controller

		for (auto element : root) {
			element->update( this );
		}
	}

	void Controller::make_text(const std::string& text, glm::vec2 pos, const glm::vec3& color, float scale, glm::vec2 size)
	{
		pos *= global_scale;
		size *= global_scale;
		scale *= global_scale;

		// Clamp to workspace limits
		size = glm::clamp(size, glm::vec2(0.f), workspace.size - pos);

		// To workspace local size
		pos -= (workspace.size - size - pos);

		TextEntity* e_text = new TextEntity(text);
		e_text->set_color(color)->set_scale(scale);
		e_text->generate_mesh();

		TextWidget* widget = new TextWidget(e_text, pos, eWidgetType::TEXT);
		root.push_back(widget);
	}

	void Controller::make_button(const std::string& signal, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture)
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
		Mesh* mesh = new Mesh();
		mesh->create_quad(size.x, size.y, color);
		e_button->set_mesh(mesh);

		ButtonWidget* widget = new ButtonWidget(signal, e_button, pos, color, size, eWidgetType::BUTTON);
		root.push_back(widget);
	}

	void Controller::make_slider(const std::string& signal, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture)
	{
		pos *= global_scale;
		size *= global_scale;

		// Clamp to workspace limits
		size = glm::clamp(size, glm::vec2(0.f), workspace.size - pos);

		// To workspace local size
		// ... doing this each frame to compute thumb position

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_track = new EntityMesh();
		Mesh * mesh = new Mesh();
		mesh->create_quad(size.x, size.y, colors::GRAY);
		e_track->set_mesh(mesh);
		
		EntityMesh* e_thumb = new EntityMesh();
		Mesh* thumb_mesh = new Mesh();
		thumb_mesh->create_quad(size.y, size.y, color);
		e_thumb->set_mesh(thumb_mesh);

		SliderWidget* widget = new SliderWidget(signal, e_track, e_thumb, pos, color, size, eWidgetType::SLIDER);
		root.push_back(widget);
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