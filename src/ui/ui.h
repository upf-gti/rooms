#pragma once

#include "includes.h"

#define RED glm::vec3(1.f, 0.f, 0.f)
#define GREEN glm::vec3(0.f, 1.f, 0.f)
#define BLUE glm::vec3(0.f, 0.f, 1.f)
#define PURPLE glm::vec3(1.f, 0.f, 1.f)

namespace ui {

	class Controller {

		struct WorkSpaceData {
			glm::vec2 size;
			uint8_t select_button;
			uint8_t root_pose;
			uint8_t hand;
			uint8_t select_hand;
		};

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

		bool make_button(glm::vec2 pos, glm::vec2 size, glm::vec3 color, const char* texture = nullptr);
	};
}