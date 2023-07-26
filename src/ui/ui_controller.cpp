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

	void Controller::process_params(glm::vec2& position, glm::vec2& size, bool skip_to_local)
	{
		position *= global_scale;
		size *= global_scale;

		// Clamp to workspace limits
		size = glm::clamp(size, glm::vec2(0.f), workspace.size - position);

		// To workspace local size
		if(!skip_to_local)
			position -= (workspace.size - size - position);
	}

	EntityMesh* Controller::make_rect(glm::vec2 pos, glm::vec2 size, const glm::vec3& color)
	{
		process_params(pos, size);

		// Render quad in local workspace position
		EntityMesh* rect = new EntityMesh();
		Mesh* mesh = new Mesh();
		mesh->create_quad(size.x, size.y, color);
		rect->set_mesh(mesh);

		Widget* widget = new Widget(rect, pos);
		root.push_back(widget);

		return rect;
	}

	void Controller::make_text(const std::string& text, glm::vec2 pos, const glm::vec3& color, float scale, glm::vec2 size)
	{
		process_params(pos, size);
		scale *= global_scale;

		TextEntity* e_text = new TextEntity(text);
		e_text->set_color(color)->set_scale(scale);
		e_text->generate_mesh();

		TextWidget* widget = new TextWidget(e_text, pos);
		root.push_back(widget);
	}

	void Controller::make_button(const std::string& signal, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture)
	{
		process_params(pos, size);

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_button = new EntityMesh();
		Mesh* mesh = new Mesh();
		mesh->create_quad(size.x, size.y);
		e_button->set_mesh(mesh);

		ButtonWidget* widget = new ButtonWidget(signal, e_button, pos, color, size);
		root.push_back(widget);
	}

	void Controller::make_slider(const std::string& signal, float default_value, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture)
	{
		process_params(pos, size, true);

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

		SliderWidget* widget = new SliderWidget(signal,e_track, e_thumb, default_value, pos, color, size);
		root.push_back(widget);
	}

	void Controller::make_color_picker(const std::string& signal, const glm::vec3& default_color, glm::vec2 pos, glm::vec2 size)
	{
		glm::vec2 offset = {0.f, size.y + 1.f};
		make_slider(signal + "_r", default_color.r, pos, size, colors::RED);
		make_slider(signal + "_g", default_color.g, pos + offset, size, colors::GREEN);
		make_slider(signal + "_b", default_color.b, pos + offset * 2.f, size, colors::BLUE);
		
		// Get color rect entity
		EntityMesh* rect = make_rect(glm::vec2(pos.x + size.x + 1.f, pos.y), glm::vec2(size.y, size.y + offset.y * 2.f), colors::WHITE);
		rect->get_mesh()->update_material_color(default_color);

		ColorPickerWidget* widget = new ColorPickerWidget(rect, default_color);
		root.push_back(widget);

		connect(signal + "_r", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.x = value;
			r->get_mesh()->update_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		connect(signal + "_g", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.y = value;
			r->get_mesh()->update_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		connect(signal + "_b", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.z = value;
			r->get_mesh()->update_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});
	}

	void Controller::connect(const std::string& name, std::variant<FuncFloat, FuncString, FuncVec2, FuncVec3> callback)
	{
		signals[name].push_back(callback);
	}
}