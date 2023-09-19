#include "ui/ui_widgets.h"
#include "ui/ui_controller.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"
#include "graphics/raymarching_renderer.h"
#include "framework/entities/entity_text.h"

namespace ui {

    Widget* Widget::current_selected = nullptr;

    Widget::Widget(EntityMesh* e, const glm::vec2& p) : entity(e), position(p)
    {
        // Bind uniforms

        auto webgpu_context = Renderer::instance->get_webgpu_context();

        uniforms.data = webgpu_context->create_buffer(sizeof(sUIData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "ui_buffer");
        uniforms.binding = 0;
        uniforms.buffer_size = sizeof(sUIData);
    }

	void Widget::add_child(Widget* child)
	{
		if (child->parent) {
			std::cerr << "Child has already a parent, remove it first!" << std::endl;
			return;
		}

		// Checks if it's already a child
		auto it = std::find(children.begin(), children.end(), child);
		if (it != children.end()) {
			std::cerr << "Widget is already one of the children!" << std::endl;
			return;
		}

		child->parent = this;
		children.push_back(child);
	}

    void Widget::set_show_children(bool value)
    {
        ButtonWidget* bw = dynamic_cast<ButtonWidget*>(this);
        if (bw && bw->is_submenu) {
            show_children = value;
            selected = value;
        }

        // Only hiding is recursive...
        if (!value)
        {
            for (auto w : children)
                w->set_show_children(value);
        }
    }

    void Widget::set_selected(bool value)
    {
        selected = value;

        // Only unselecting is recursive...
        if (!value)
        {
            for (auto w : children)
                w->set_selected(value);
        }
    }

	void Widget::render()
	{
		entity->render();

		if (!show_children)
			return;

		for (auto c : children)
			c->render();
	}

	void Widget::update(Controller* controller)
	{
		entity->set_model(controller->get_matrix());
		entity->translate(glm::vec3(position.x, position.y, -1e-3f - priority * 1e-3f));

		if (!show_children)
			return;

		for (auto c : children)
			c->update( controller );
	}

    /*
    *   Widget Group
    */

    WidgetGroup::WidgetGroup(EntityMesh* e, const glm::vec2& p, float n) : Widget(e, p) {

        show_children = true;
        type = eWidgetType::GROUP;

        ui_data.num_group_items = n;

        auto webgpu_context = Renderer::instance->get_webgpu_context();
        bind_group = webgpu_context->create_bind_group({ &uniforms }, Shader::get("data/shaders/mesh_ui.wgsl"), 2);
    }

	/*
	*	Button
	*/

    ButtonWidget::ButtonWidget(const std::string& sg, EntityMesh* e, const glm::vec2& p, const Color& c, const glm::vec2& s)
        : Widget(e, p), signal(sg), size(s), color(c) {

        type = eWidgetType::BUTTON;

        auto webgpu_context = Renderer::instance->get_webgpu_context();
        bind_group = webgpu_context->create_bind_group({ &uniforms }, Shader::get("data/shaders/mesh_texture_ui.wgsl"), 2);
    }

	void ButtonWidget::update(Controller* controller)
	{
		Widget::update(controller);

		/*
		*	Manage intersection
		*/

		const WorkSpaceData& workspace = controller->get_workspace();

		uint8_t hand = workspace.hand;
		uint8_t select_hand = workspace.select_hand;
		uint8_t pose = workspace.root_pose;

		// Ray
		glm::vec3 ray_origin = Input::get_controller_position(select_hand, pose);
		glm::mat4x4 select_hand_pose = Input::get_controller_pose(select_hand, pose);
		glm::vec3 ray_direction = get_front(select_hand_pose);

		// Quad
		glm::vec3 quad_position = entity->get_translation();
		glm::quat quad_rotation = glm::quat_cast(controller->get_matrix());

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

        entity->set_material_color(/*is_pressed ? colors::GRAY : */color);
		
        if (was_pressed)
        {
            selected = !selected;

			controller->emit_signal(signal, color);

            if (selected && is_color_button)
            {
                if(current_selected) current_selected->selected = false;
                current_selected = this;
            }
        }

        // Update uniforms
        ui_data.is_hovered = hovered ? 1.f : 0.f;
        ui_data.is_selected = selected ? 1.f : 0.f;
        ui_data.is_color_button = is_color_button ? 1.f : 0.f;
	}

	/*
	*	Slider
	*/

	void SliderWidget::render()
	{
		Widget::render();
		thumb_entity->render();
	}

	void SliderWidget::update(Controller* controller)
	{
		const WorkSpaceData& workspace = controller->get_workspace();

		// We need to store this position before converting to local size
		glm::vec2 pos = position;
		glm::vec2 thumb_size = { size.y, size.y };

		// No value assigned
		if (current_slider_pos == -1)
		{
			max_slider_pos = (size.x * 2.f - thumb_size.x * 2.f) / controller->global_scale;
			current_slider_pos = current_value * max_slider_pos;
		}

		float thumb_pos = current_slider_pos * controller->global_scale - workspace.size.x + thumb_size.x + pos.x * 2.f;

		// To workspace local size
		pos -= (workspace.size - size - pos);

		/*
		*	Update elements
		*/

		entity->set_model(controller->get_matrix());
		entity->translate(glm::vec3(pos.x, pos.y, -1e-3f));

		thumb_entity->set_model(controller->get_matrix());
		thumb_entity->translate(glm::vec3(thumb_pos, pos.y, -2e-3f));

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
		glm::vec3 quad_position = thumb_entity->get_translation();
		glm::vec3 slider_quad_position = entity->get_translation();
		glm::quat quad_rotation = glm::quat_cast(controller->get_matrix());

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

		if (is_pressed)
		{
			current_slider_pos = glm::clamp((intersection.x + size.x - thumb_size.x) / controller->global_scale, 0.f, max_slider_pos);
			current_value = glm::clamp(current_slider_pos / max_slider_pos, 0.f, 1.f);
			controller->emit_signal(signal, current_value);
		}
	}
}
