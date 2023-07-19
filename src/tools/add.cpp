#include "add.h"

#include "utils.h"

void AdditionTool::initialize() {}
void AdditionTool::clean() {}

std::vector<sEdit> AdditionTool::use() {
	std::vector<sEdit> list;

	list.push_back({
#ifdef XR_SUPPORT
			.position = Input::get_controller_position(HAND_LEFT),
#else
			.position = glm::vec3(random_f(), random_f(), random_f()),
#endif
			.primitive = selected_primitive,
			.color = glm::vec3(random_f(), random_f(), random_f()),
			.operation = (is_smooth) ? OP_SMOOTH_UNION : OP_UNION,
			.size = size,
			.radius = radius
		});

	return list;
}

void AdditionTool::update(float delta_tim) {}
void AdditionTool::render_scene() {}
void AdditionTool::render_ui() {}