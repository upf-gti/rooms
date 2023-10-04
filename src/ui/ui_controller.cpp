#include "ui/ui_controller.h"
#include "ui/ui_widgets.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/renderers/rooms_renderer.h"
#include "framework/entities/entity_text.h"

#include "framework/scene/parse_scene.h"

namespace ui {

    std::map<std::string, Widget*> Controller::all_widgets;

    float current_number_of_group_widgets; // store to make sure everything went well

	void Controller::set_workspace(glm::vec2 _workspace_size, uint8_t _select_button, uint8_t _root_pose, uint8_t _hand, uint8_t _select_hand)
	{
        renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

		global_scale = 0.001f;

		workspace = {
			.size = _workspace_size * global_scale,
			.select_button = _select_button,
			.root_pose = _root_pose,
			.hand = _hand,
			.select_hand = _select_hand
		};

        root = new Widget();

        raycast_pointer = parse_scene("data/meshes/raycast.obj");

        // Debug
        if (render_background)
        {
            background = new EntityMesh();
            background->set_material_shader(Shader::get("data/shaders/mesh_color.wgsl"));
            Mesh* mesh = new Mesh();
            mesh->create_quad();
            background->set_mesh(mesh);
        }
	}

	bool Controller::is_active()
	{
        return workspace.hand == HAND_RIGHT ||
            Input::get_grab_value(workspace.hand) > 0.5f;
	}

	void Controller::render()
	{
		if (!enabled || !is_active()) return;

        if (render_background) background->render();

		for (auto widget : root->children) {
			widget->render();
		}

        if(workspace.hand == HAND_LEFT)
		    raycast_pointer->render();
	}

	void Controller::update(float delta_time)
	{
		if (!enabled || !is_active()) return;

		uint8_t hand = workspace.hand;
		uint8_t pose = workspace.root_pose;

		// Update raycast helper

        if (hand == HAND_LEFT)
        {
		    glm::mat4x4 raycast_transform = Input::get_controller_pose(workspace.select_hand, pose);
		    raycast_pointer->set_model(raycast_transform);
		    raycast_pointer->rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
		    raycast_pointer->scale(glm::vec3(0.1f));
        }

		// Update workspace

		glm::mat4x4 workspace_transform = 
        global_transform = Input::get_controller_pose(hand, pose);
        global_transform = glm::rotate(global_transform, glm::radians(120.f), glm::vec3(1.f, 0.f, 0.f));

        if (pose == POSE_GRIP)
            global_transform = glm::rotate(global_transform, glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));

        if (render_background)
        {
            background->set_model(global_transform);
            background->translate(glm::vec3(0.f, 0.f, 1e-3f));
            background->scale(glm::vec3(workspace.size.x, workspace.size.y, 1.f));
            
        }

		// Update widgets using this controller

		for (auto element : root->children) {
			element->update( this );
		}

        // Update labels

        for (auto& it : get_widgets())
        {
            ui::LabelWidget* widget = static_cast<ui::LabelWidget*>(it.second);

            if (!widget->active || widget->type != ui::LABEL)
                continue;

            if (widget->button != -1 && Input::was_button_pressed(widget->button))
            {
                widget->selected = !widget->selected;
                ui::TextWidget* text_label = static_cast<ui::TextWidget*>(get_widget_from_name("text@" + it.first));
                ((TextEntity*)text_label->entity)->set_text(widget->selected ? widget->subtext : widget->text);
            }
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
    glm::vec2 Controller::compute_position()
    {
        float x, y;

        if (group_opened)
        {
            x = last_layout_pos.x + g_iterator * BUTTON_SIZE + g_iterator * X_GROUP_MARGIN;
            y = last_layout_pos.y;
            g_iterator++;
        }
        else
        {
            x = layout_iterator.x;
            y = layout_iterator.y * BUTTON_SIZE + (layout_iterator.y + 1.f) * Y_MARGIN;
            layout_iterator.x += BUTTON_SIZE + X_MARGIN;
            last_layout_pos = { x, y };
        }

        return { x, y };
    }

	void Controller::append_widget(Widget* widget, const std::string& name)
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

        widgets[name] = widget;
        all_widgets[name] = widget;
	}

	Widget* Controller::make_rect(glm::vec2 pos, glm::vec2 size, const Color& color)
	{
		process_params(pos, size);

		// Render quad in local workspace position
		EntityMesh* rect = new EntityMesh();
		rect->set_material_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh* mesh = new Mesh();
		mesh->create_quad();
		rect->set_mesh(mesh);
        rect->set_material_color(color);

		Widget* widget = new Widget(rect, pos, size);
		append_widget(widget, "ui_rect");
		return widget;
	}

	Widget* Controller::make_text(const std::string& text, const std::string& alias, glm::vec2 pos, const Color& color, float scale, glm::vec2 size)
	{
		process_params(pos, size);
		scale *= global_scale;

		TextEntity* e_text = new TextEntity(text);
		e_text->set_material_color(color);
		e_text->set_scale(scale);
		e_text->generate_mesh();

		TextWidget* widget = new TextWidget(e_text, pos);
		append_widget(widget, alias);
		return widget;
	}

    Widget* Controller::make_label(const json& j)
    {
        std::string text = j["name"];
        std::string alias = j.value("alias", text);

        static int num_labels = 0;

        // World attributes
        float workspace_width = workspace.size.x / global_scale;
        float margin = 2.f;

        // Text follows after icon (right)
        glm::vec2 pos = { LABEL_BUTTON_SIZE + 4.f, num_labels * LABEL_BUTTON_SIZE + (num_labels + 1.f) * margin };
        glm::vec2 size = glm::vec2(workspace_width, LABEL_BUTTON_SIZE);

        Widget* text_widget = make_text(text, "text@" + alias, pos, colors::WHITE, 12.f);
        text_widget->priority = -1;

        // Icon 
        EntityMesh* m_icon = new EntityMesh();
        Mesh* mesh = new Mesh();
        if (j.count("texture") > 0)
        {
            std::string texture = j["texture"];
            m_icon->set_material_diffuse(Texture::get(texture));
        }

        // Icon goes to the left of the workspace
        pos = { 
            -workspace_width * 0.5f + LABEL_BUTTON_SIZE * 0.5f,
            num_labels * LABEL_BUTTON_SIZE + (num_labels + 1.f) * margin
        };

        process_params(pos, size);

        mesh->create_quad();
        m_icon->set_mesh(mesh);
        m_icon->set_material_shader(Shader::get("data/shaders/mesh_texture.wgsl"));

        LabelWidget* label_widget = new LabelWidget(text, m_icon, pos, glm::vec2(size.y, size.y));
        label_widget->button = j.value("button", -1);
        label_widget->subtext = j.value("subtext", "");

        append_widget(label_widget, alias);

        num_labels++;

        return label_widget;
    }

	Widget* Controller::make_button(const json& j)
	{
        std::string signal = j["name"];

        // World attributes
        glm::vec2 pos = compute_position();
        glm::vec2 size = glm::vec2(BUTTON_SIZE);

		glm::vec2 _pos = pos;
		glm::vec2 _size = size;

		process_params(pos, size);

		/*
		*	Create button entity and set transform
		*/

        std::string texture = j["texture"];
        std::string shader = "data/shaders/mesh_texture_ui.wgsl";

        if (j.count("shader"))
            shader = j["shader"];

		// Render quad in local workspace position
		EntityMesh* e_button = new EntityMesh();
		e_button->set_material_shader(Shader::get(shader));
		e_button->set_material_diffuse(Texture::get(texture));

		Mesh* mesh = new Mesh();
		mesh->create_quad();
        mesh->set_alias(signal);
		e_button->set_mesh(mesh);

        const bool allow_toggle = j.value("allow_toggle", false);
        const bool is_color_button = j.count("color") > 0;
        Color color = is_color_button ? load_vec4(j["color"]) : colors::WHITE;

		ButtonWidget* widget = new ButtonWidget(signal, e_button, pos, size, color);
        widget->is_color_button = is_color_button;
        widget->is_unique_selection = j.value("unique_selection", false);

        if( group_opened )
            widget->priority = 1;

        if (is_color_button || widget->is_unique_selection || allow_toggle)
        {
            bind(signal, [widget = widget, allow_toggle](const std::string& signal, void* button) {
                // Unselect siblings
                Widget* parent = widget->parent;
                const bool last_value = widget->selected;
                if (!allow_toggle)
                {
                    for (auto w : parent->children)
                        w->set_selected(false);
                }
                widget->set_selected(allow_toggle ? !last_value : true);
            });
        }

        widget->m_layer = static_cast<uint8_t>(layout_iterator.y);

		append_widget(widget, signal);

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
		e_track->set_material_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh * mesh = new Mesh();
		mesh->create_quad();
		e_track->set_mesh(mesh);
        e_track->set_material_color(colors::GRAY);
		
		EntityMesh* e_thumb = new EntityMesh();
		e_thumb->set_material_shader(Shader::get("data/shaders/mesh_color.wgsl"));
		Mesh* thumb_mesh = new Mesh();
		thumb_mesh->create_quad();
		e_thumb->set_mesh(thumb_mesh);
        e_thumb->set_material_color(color);

		SliderWidget* widget = new SliderWidget(signal,e_track, e_thumb, default_value, pos, color, size);
		append_widget(widget, signal);
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
		rect->set_material_color( default_color );

		ColorPickerWidget* widget = new ColorPickerWidget(rect, default_color);
		append_widget(widget, signal);

		bind(signal + "_r", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.x = value;
			r->set_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		bind(signal + "_g", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.y = value;
			r->set_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		bind(signal + "_b", [this, signal, w = widget, r = rect](const std::string& s, float value) {
			w->rect_color.z = value;
			r->set_material_color(w->rect_color);
			emit_signal(signal, w->rect_color);
		});

		return widget;
	}

	void Controller::make_submenu(Widget* widget, const std::string& name)
	{
        static_cast<ButtonWidget*>(widget)->is_submenu = true;

        // Visibility callback...
		bind(name, [widget = widget](const std::string& signal, void* button) {

            Widget* parent = widget->parent;
            if (parent->type == GROUP)
                parent = parent->parent;

            const bool last_value = widget->show_children;

            for (auto w : parent->children)
                w->set_show_children(false);

            widget->set_show_children(!last_value);
		});

        layout_iterator.x = 0.f;
        layout_iterator.y = widget->m_layer + 1.f;

        // Update last layout pos
        float x = 0.f;
        float y = layout_iterator.y * BUTTON_SIZE + (layout_iterator.y + 1.f) * Y_MARGIN;
        last_layout_pos = { x, y };

        // Set as new parent...
        parent_queue.push_back(widget);
	}

    void Controller::close_submenu()
    {
        parent_queue.pop_back();
    }

    Widget* Controller::make_group(const std::string& group_name, float number_of_widgets, const Color& color)
    {
        current_number_of_group_widgets = number_of_widgets;

        // World attributes
        glm::vec2 pos = compute_position() - 4.f;
        glm::vec2 size = glm::vec2(
            BUTTON_SIZE * number_of_widgets + (number_of_widgets - 1.f) * X_GROUP_MARGIN + 8.f,
            BUTTON_SIZE + 8.f
        );

        glm::vec2 _size = size;

        process_params(pos, size);

        EntityMesh* e = new EntityMesh();
        e->set_material_shader(Shader::get("data/shaders/mesh_ui.wgsl"));
        Mesh* mesh = new Mesh();
        mesh->create_quad();
        mesh->set_alias(group_name);
        e->set_mesh(mesh);
        e->set_material_color(color);

        WidgetGroup* group = new WidgetGroup(e, pos, size, number_of_widgets);
        append_widget(group, group_name);

        parent_queue.push_back(group);
        group_opened = true;
        layout_iterator.x += _size.x - (BUTTON_SIZE + X_GROUP_MARGIN);

        return group;
    }

    void Controller::close_group()
    {
        assert(g_iterator == current_number_of_group_widgets && "Num Widgets in group does not correspond");

        // Clear group info
        parent_queue.pop_back();
        group_opened = false; 
        g_iterator = 0.f;
    }

	void Controller::bind(const std::string& name, SignalType callback)
	{
		signals[name].push_back(callback);
	}

    Widget* Controller::get(const std::string& alias)
    {
        if (all_widgets.count(alias)) return all_widgets[alias];
        return nullptr;
    }

    Widget* Controller::get_widget_from_name(const std::string& alias)
    {
        if(widgets.count(alias)) return widgets[alias];
        return nullptr;
    }

    void Controller::load_layout(const std::string& filename)
    {
        const json& j = load_json(filename);
        mjson = j;
        float group_elements_pending = -1;

        float width = j["width"];
        float height = j["height"];
        set_workspace({ width, height });

        std::function<void(const json&)> read_element = [&](const json& j) {

            std::string name = j["name"];
            std::string type = j["type"];

            if (type == "group")
            {
                assert(j.count("nitems") > 0);
                float nitems = j["nitems"];
                group_elements_pending = nitems;

                glm::vec4 color;
                if (j.count("color")) {
                    color = load_vec4(j["color"]);
                }
                else {
                    color = colors::GRAY;
                }

                Widget* group = make_group(name, nitems, color);
            }
            else if (type == "button")
            {
                ButtonWidget* widget = (ButtonWidget*)make_button(j);
                group_elements_pending--;

                widget->selected = j.value("selected", false);

                if (group_elements_pending == 0.f)
                {
                    close_group();
                    group_elements_pending = -1;
                }
            }
            else if (type == "label")
            {
                make_label(j);
            }
            else if (type == "submenu")
            {
                Widget* parent = get_widget_from_name(name);

                if (!parent) {
                    assert(0);
                    std::cerr << "Can not find parent button with name " << name << std::endl;
                    return;
                }

                make_submenu(parent, name);

                assert(j.count("children") > 0);
                auto& _subelements = j["children"];
                for (auto& sub_el : _subelements) {
                    read_element(sub_el);
                }

                close_submenu();
            }
            };

        auto& _elements = j["elements"];
        for (auto& el : _elements) {
            read_element(el);
        }
    }

    void Controller::change_list_layout(const std::string& list_name)
    {
        if (mjson.count("lists") == 0) {
            std::cerr << "Controller doesn't have layout lists..." << std::endl;
            return;
        }

        // Disable all widgets
        for (auto& w : widgets) {
            w.second->active = false;
        }

        // Enable only widgets in list...
        const json& lists = mjson["lists"];
        if (lists.count(list_name) == 0) {
            std::cerr << "Controller doesn't have a layout list named '" << list_name << "'" << std::endl;
            return;
        }

        for (auto& it : lists[list_name]) {
            const std::string& name = it;
            auto widget = get_widget_from_name(name);
            widget->active = true;

            // Display also its text...
            if (widget->type == LABEL)
            {
                widget = get_widget_from_name("text@" + name);
                widget->active = true;
            }
        }
    }
}
