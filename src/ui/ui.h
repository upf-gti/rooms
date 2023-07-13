#pragma once

#include "includes.h"

class RaymarchingRenderer;

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

	public:

		/*
		*	Select button: XR Buttons
		*	Root pose: AIM, GRIP
		*/

		void set_workspace(glm::vec2 _workspace_size, uint8_t _select_button = 0, uint8_t _root_pose = 0, uint8_t _hand = 0, uint8_t _select_hand = 1);
		void render( RaymarchingRenderer* _renderer);

		/*
		*	Widgets
		*/

		bool make_button(glm::vec2 pos, glm::vec2 size, const char* texture = nullptr);
	};
}