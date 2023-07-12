#pragma once

#include "includes.h"

namespace ui {

	class Controller {

		struct WorkSpaceData {
			glm::vec2 size;
			uint8_t select_button;
			uint8_t root_pose;
		};

		WorkSpaceData workspace;

	public:

		/*
		*	Select button: XR Buttons
		*	Root pose: AIM, GRIP
		*/

		void set_workspace(glm::vec2 _workspace_size, uint8_t _select_button = 0, uint8_t _root_pose = 0);
		void render();

		/*
		*	Widgets
		*/

		bool make_button(glm::vec2 pos, glm::vec2 size, const char* texture = nullptr);
	};
}