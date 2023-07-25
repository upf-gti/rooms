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

		Widget(EntityMesh* e, const glm::vec2& p, uint8_t t) 
			: entity(e), position(p), type(t) {};

		EntityMesh* entity = nullptr;
		glm::vec2 position;
		uint8_t type = eWidgetType::NONE;

		virtual void render();
		virtual void update(Controller* controller);
	};

	class TextWidget : public Widget {
	public:

		TextWidget(EntityMesh* e, const glm::vec2& pos, uint8_t t)
			: Widget(e, pos, t) {};
	};

	class ButtonWidget : public Widget {
	public:

		glm::vec2 size;
		std::string signal;
		Color color;

		ButtonWidget(const std::string& sg, EntityMesh* e, const glm::vec2& p, const glm::vec3& c, const glm::vec2& s, uint8_t t)
			: Widget(e, p, t), signal(sg), size(s), color(c) {};

		virtual void update(Controller* controller) override;
	};

	class SliderWidget : public ButtonWidget {
	public:

		EntityMesh* thumb_entity = nullptr;

		float current_slider_pos = 0.f;
		float max_slider_pos = -1.f;

		SliderWidget(const std::string& sg, EntityMesh* tr, EntityMesh* th, const glm::vec2& p, const glm::vec3& c, const glm::vec2& s, uint8_t t)
			: ButtonWidget(sg, tr, p, c, s, t), thumb_entity(th) {};

		virtual void render() override;
		virtual void update(Controller* controller) override;
	};
}