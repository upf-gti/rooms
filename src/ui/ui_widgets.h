#pragma once

#include "framework/colors.h"
#include "graphics/webgpu_context.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

class EntityMesh;

namespace ui {

	class Controller;

    struct sUIData {
        float num_group_items   = 2;
        float is_selected       = 0.f;
        float is_hovered        = 0.f;
        float is_color_button   = 0.f;
    };

	enum eWidgetType {
		NONE,
		TEXT,
		BUTTON,
		SLIDER,
        GROUP
	};

	class Widget {
	public:

        Widget() {}
        Widget(EntityMesh* e, const glm::vec2& p);

        uint8_t type = eWidgetType::NONE;
		Widget* parent = nullptr;

        static Widget* current_selected;

		std::vector<Widget*> children;
		bool show_children  = false;
        bool selected       = false;

		EntityMesh* entity = nullptr;
		glm::vec2   position;
        uint8_t     m_layer = 0;
		int         priority = 0;

        sUIData         ui_data;
        WGPUBindGroup   bind_group = nullptr;
        Uniform         uniforms;

		void add_child(Widget* child);

        void set_show_children(bool value);
        void set_selected(bool value);

		virtual void render();
		virtual void update(Controller* controller);
	};

    class WidgetGroup : public Widget {
    public:

        WidgetGroup(EntityMesh* e, const glm::vec2& p, float number_of_widgets);
    };

	class TextWidget : public Widget {
	public:

		TextWidget(EntityMesh* e, const glm::vec2& pos)
			: Widget(e, pos) {
			type = eWidgetType::TEXT;
		}
	};

	class ButtonWidget : public Widget {
	public:

		glm::vec2 size;
		std::string signal;

        Color color;
        bool is_color_button = false;

        bool is_submenu = false;

        ButtonWidget(const std::string& sg, EntityMesh* e, const glm::vec2& p, const Color& c, const glm::vec2& s);

		virtual void update(Controller* controller) override;
	};

	class SliderWidget : public ButtonWidget {
	public:

		EntityMesh* thumb_entity = nullptr;

		float current_value = 0.f;
		float current_slider_pos = -1.f;
		float max_slider_pos = -1.f;

		SliderWidget(const std::string& sg, EntityMesh* tr, EntityMesh* th, float v, const glm::vec2& p, const Color& c, const glm::vec2& s)
			: ButtonWidget(sg, tr, p, c, s), thumb_entity(th), current_value(v) {
			type = eWidgetType::SLIDER;
		}

		virtual void render() override;
		virtual void update(Controller* controller) override;
	};

	class ColorPickerWidget : public Widget {
	public:

		Color rect_color;

		ColorPickerWidget(EntityMesh* rect, const Color& color) : rect_color(color) {
			entity = rect;
		}

		virtual void render() override {};
		virtual void update(Controller* controller) override {};
	};
}
