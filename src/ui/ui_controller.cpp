#include "ui/ui_controller.h"
#include "ui/ui_widgets.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"
#include "framework/entities/entity_text.h"

namespace ui {

    float current_number_of_group_widgets; // store to make sure everything went well

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

        root = new Widget();

		raycast_pointer = new EntityMesh();
		raycast_pointer->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		raycast_pointer->set_mesh(Mesh::get("data/meshes/raycast.obj"));
	}

	bool Controller::is_active()
	{
		return Input::get_grab_value(workspace.hand) > 0.5f;
	}

	void Controller::render()
	{
		if (!enabled || !is_active()) return;

		for (auto widget : root->children) {
			widget->render();
		}

		raycast_pointer->render();
	}

	void Controller::update(float delta_time)
	{
		if (!enabled || !is_active()) return;

		uint8_t hand = workspace.hand;
		uint8_t pose = workspace.root_pose;

		// Update raycast helper

		glm::mat4x4 raycast_transform = Input::get_controller_pose(workspace.select_hand, pose);
		raycast_pointer->set_model(raycast_transform);
		raycast_pointer->rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
		raycast_pointer->scale(glm::vec3(0.1f));

		// Update workspace

		glm::mat4x4 workspace_transform = 
        global_transform = Input::get_controller_pose(hand, pose);
        global_transform = glm::rotate(global_transform, glm::radians(120.f), glm::vec3(1.f, 0.f, 0.f));

		// Update widgets using this controller

		for (auto element : root->children) {
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

    // Gets next button position (applies margin)
    const glm::vec2& Controller::compute_position()
    {
        float x, y;

        if (group_opened)
        {
            x = last_layout_pos.x + g_iterator * BUTTON_SIZE + g_iterator * 4.f;
            y = last_layout_pos.y;
            g_iterator++;
        }
        else
        {
            x = layout_iterator.x * BUTTON_SIZE + (layout_iterator.x + 1.f) * 8.f;
            y = layout_iterator.y * BUTTON_SIZE + (layout_iterator.y + 1.f) * 12.f;
            layout_iterator.x++;
            last_layout_pos = { x, y };
        }

        return { x, y };
    }

	void Controller::append_widget(Widget* widget)
	{
        if (parent_queue.size())
        {
            Widget* active_submenu = parent_queue.back();
			active_submenu->add_child(widget);
        }
		else
		{
			root->add_child(widget);
		}
	}

	Widget* Controller::make_rect(glm::vec2 pos, glm::vec2 size, const Color& color)
	{
		process_params(pos, size);

		// Render quad in local workspace position
		EntityMesh* rect = new EntityMesh();
		rect->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh* mesh = new Mesh();
		mesh->create_quad(size.x, size.y, color);
		rect->set_mesh(mesh);

		Widget* widget = new Widget(rect, pos);
		append_widget(widget);
		return widget;
	}

	Widget* Controller::make_text(const std::string& text, glm::vec2 pos, const Color& color, float scale, glm::vec2 size)
	{
		process_params(pos, size);
		scale *= global_scale;

		TextEntity* e_text = new TextEntity(text);
		e_text->set_shader(Shader::get("data/shaders/sdf_fonts.wgsl"));
		e_text->set_color(color);
		e_text->set_scale(scale);
		e_text->generate_mesh();

		TextWidget* widget = new TextWidget(e_text, pos);
		append_widget(widget);
		return widget;
	}

	Widget* Controller::make_button(const std::string& signal, const char* texture, const char* texture_selected, const Color& color)
	{
        // World attributes
        glm::vec2 pos = compute_position();
        glm::vec2 size = glm::vec2(BUTTON_SIZE);

		glm::vec2 _pos = pos;
		glm::vec2 _size = size;

		process_params(pos, size);

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_button = new EntityMesh();
		e_button->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh* mesh = new Mesh();
		if (texture)
		{
			e_button->set_shader(Shader::get("data/shaders/mesh_texture.wgsl"));
			mesh->set_texture(Texture::get(texture));
		}
		else
		{
			// Text: Use for debug  only
			Widget* text = make_text(signal, _pos, colors::BLACK, _size.x * 0.2f);
			text->priority = 1;
		}
		mesh->create_quad(size.x, size.y);
		e_button->set_mesh(mesh);

		ButtonWidget* widget = new ButtonWidget(signal, e_button, pos, color, size);

        if( group_opened )
            widget->priority = 1;

        widget->m_layer = layout_iterator.y;

		append_widget(widget);

        // Hide submenus of siblings
        connect(signal, [widget = widget, childs = &widget->parent->children](const std::string& signal, float value) {
            for (auto w : *childs)
                w->set_show_children(false);
        });

        /*if (texture && texture_selected)
        {
            std::string tex(texture);
            std::string tex_selected(texture_selected);

            connect(signal, [widget = widget, tex, tex_selected, mesh = mesh](const std::string& signal, float value) {
                widget->selected = !widget->selected;
                mesh->set_texture(Texture::get(widget->selected ? tex_selected.c_str() : tex.c_str()));
            });
        }*/

		return widget;
	}

	Widget* Controller::make_slider(const std::string& signal, float default_value, glm::vec2 pos, glm::vec2 size, const Color& color, const char* texture)
	{
		process_params(pos, size, true);

		/*
		*	Create button entity and set transform
		*/

		// Render quad in local workspace position
		EntityMesh* e_track = new EntityMesh();
		e_track->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh * mesh = new Mesh();
		mesh->create_quad(size.x, size.y, colors::GRAY);
		e_track->set_mesh(mesh);
		
		EntityMesh* e_thumb = new EntityMesh();
		e_thumb->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh* thumb_mesh = new Mesh();
		thumb_mesh->create_quad(size.y, size.y, color);
		e_thumb->set_mesh(thumb_mesh);

		SliderWidget* widget = new SliderWidget(signal,e_track, e_thumb, default_value, pos, color, size);
		append_widget(widget);
		return widget;
	}

	Widget* Controller::make_color_picker(const std::string& signal, const Color& default_color, glm::vec2 pos, glm::vec2 size)
	{
		glm::vec2 offset = {0.f, size.y + 1.f};
		make_slider(signal + "_r", default_color.r, pos, size, colors::RED);
		make_slider(signal + "_g", default_color.g, pos + offset, size, colors::GREEN);
		make_slider(signal + "_b", default_color.b, pos + offset * 2.f, size, colors::BLUE);
		
		// Get color rect entity
		Widget* widget_rect = make_rect(glm::vec2(pos.x + size.x + 1.f, pos.y), glm::vec2(size.y, size.y + offset.y * 2.f), colors::WHITE);
		EntityMesh* rect = widget_rect->entity;
		rect->set_color( default_color );

		ColorPickerWidget* widget = new ColorPickerWidget(rect, default_color);
		append_widget(widget);

		connect(signal + "_r", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.x = value;
			r->set_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		connect(signal + "_g", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.y = value;
			r->set_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		connect(signal + "_b", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.z = value;
			r->set_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		return widget;
	}

	void Controller::make_submenu(Widget* parent, const std::string& name)
	{
        static_cast<ButtonWidget*>(parent)->is_submenu = true;

        // Visibility callback...
		connect(name, [widget = parent, childs = &parent->parent->children](const std::string& signal, float value) {
            for (auto w : *childs)
                w->set_show_children(false);
            widget->set_show_children(!widget->show_children);
		});

        layout_iterator.x = 0.f;
        layout_iterator.y = parent->m_layer + 1.f;

        // Update last layout pos
        float x = layout_iterator.x * BUTTON_SIZE + (layout_iterator.x + 1.f) * 8.f;
        float y = layout_iterator.y * BUTTON_SIZE + (layout_iterator.y + 1.f) * 12.f;
        last_layout_pos = { x, y };

        // Set as new parent...
        parent_queue.push_back(parent);
	}

    void Controller::close_submenu()
    {
        parent_queue.pop_back();
    }

    void Controller::make_group(float number_of_widgets, const Color& color)
    {
        current_number_of_group_widgets = number_of_widgets;

        // World attributes
        glm::vec2 pos = compute_position() - 4.f;
        glm::vec2 size = glm::vec2(
            BUTTON_SIZE * number_of_widgets + (number_of_widgets - 1.f) * 4.f + 8.f,
            BUTTON_SIZE + 8.f
        );

        process_params(pos, size);

        EntityMesh* e = new EntityMesh();
        e->set_shader(Shader::get("data/shaders/mesh_ui.wgsl"));
        Mesh* mesh = new Mesh();
        mesh->create_quad(size.x, size.y, color);
        e->set_mesh(mesh);

        Widget* group = new Widget(e, pos);
        group->show_children = true;
        group->type = GROUP;
        append_widget(group);

        parent_queue.push_back(group);
        group_opened = true;
        layout_iterator.x += (number_of_widgets - 1.f);
    }

    void Controller::close_group()
    {
        assert(g_iterator == current_number_of_group_widgets && "Num Widgets in group does not correspond");

        // Clear group info
        parent_queue.pop_back();
        group_opened = false; 
        g_iterator = 0.f;
    }

	void Controller::connect(const std::string& name, SignalType callback)
	{
		signals[name].push_back(callback);
	}
}
