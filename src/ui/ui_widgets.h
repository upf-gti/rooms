#pragma once

#include "framework/colors.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

class EntityMesh;

namespace ui {

	class Controller;

	enum eWidgetType {
		NONE,
		TEXT,
		BUTTON,
		SLIDER
	};

	class Widget {
	public:

		Widget() {};
		Widget(EntityMesh* e, const glm::vec2& p) 
			: entity(e), position(p) {};

		EntityMesh* entity = nullptr;
		glm::vec2 position;
		uint8_t type = eWidgetType::NONE;
		uint8_t priority = 0;

		virtual void render();
		virtual void update(Controller* controller);
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

		ButtonWidget(const std::string& sg, EntityMesh* e, const glm::vec2& p, const glm::vec3& c, const glm::vec2& s)
			: Widget(e, p), signal(sg), size(s), color(c) {
			type = eWidgetType::BUTTON;
		}

		virtual void update(Controller* controller) override;
	};

	class SliderWidget : public ButtonWidget {
	public:

		EntityMesh* thumb_entity = nullptr;

		float current_value = 0.f;
		float current_slider_pos = -1.f;
		float max_slider_pos = -1.f;

		SliderWidget(const std::string& sg, EntityMesh* tr, EntityMesh* th, float v, const glm::vec2& p, const glm::vec3& c, const glm::vec2& s)
			: ButtonWidget(sg, tr, p, c, s), thumb_entity(th), current_value(v) {
			type = eWidgetType::SLIDER;
		}

		virtual void render() override;
		virtual void update(Controller* controller) override;
	};

	class ColorPickerWidget : public Widget {
	public:

		glm::vec3 rect_color;

		ColorPickerWidget(EntityMesh* rect, const glm::vec3& color) : rect_color(color) {
			entity = rect;
		}

		virtual void render() override {};
		virtual void update(Controller* controller) override {};
	};
}