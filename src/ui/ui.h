#pragma once

#include "framework/colors.h"

namespace ui {

	struct ButtonColorData {
		Color base_color = colors::WHITE;
		Color hover_color = colors::WHITE;
		Color active_color = colors::WHITE;

		const char* texture = nullptr;
	};

	struct WorkSpaceData {
		glm::vec2 size;
		uint8_t select_button;
		uint8_t root_pose;
		uint8_t hand;
		uint8_t select_hand;
	};

	class Controller {

		WorkSpaceData workspace;
		glm::mat4x4 global_transform;

	public:

		/*
		*	Select button: XR Buttons
		*	Root pose: AIM, GRIP
		*/

		void set_workspace(glm::vec2 _workspace_size, uint8_t _select_button = 0, uint8_t _root_pose = 0, uint8_t _hand = 0, uint8_t _select_hand = 1);
		void render();
		void update(float delta_time);

		/*
		*	Widgets
		*/

		bool make_button(glm::vec2 pos, glm::vec2 size, const ButtonColorData& data);
	};
}