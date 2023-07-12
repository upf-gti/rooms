#include "ui/ui.h"
#include "utils.h"
#include "framework/intersections.h"
#include "framework/input.h"

namespace ui {

	void Controller::set_workspace(glm::vec2 _workspace_size, uint8_t _select_button, uint8_t _root_pose)
	{
		workspace = {
			.size = _workspace_size,
			.select_button = _select_button,
			.root_pose = _root_pose
		};
	}

	void Controller::render()
	{
		// Mirror
		if (make_button({ 0, 0 }, { 128, 128 })) {

			std::cout << "Button pressed!" << std::endl;

			// MirrorTool.emit('click');
		}
	}

	bool Controller::make_button(glm::vec2 pos, glm::vec2 size, const char* texture) {

		// Manage intersection

		bool hovered = false; // Intersection::ray_quad();
		bool pressed = hovered && Input::is_button_pressed( workspace.select_button );

		// Render button
		// ...

		return pressed;
	}
}