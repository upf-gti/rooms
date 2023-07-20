#include "sausage.h"

#include "utils.h"

void SausageTool::initialize() {}
void SausageTool::clean() {}

std::vector<sEdit> SausageTool::use() {
	std::vector<sEdit> list;

#ifdef XR_SUPPORT
	glm::vec3 edit_origin = Input::get_controller_position(HAND_LEFT);
#else
	glm::vec3 edit_origin = glm::vec3(random_f(), random_f(), random_f());
#endif


	list.push_back({
			.position = edit_origin,
			.primitive = selected_primitive,
			.color = glm::vec3(random_f(), random_f(), random_f()),
			.operation = (smooth_mode) ? OP_SMOOTH_UNION : OP_UNION,
			.size = second_position,
			.radius = radius
		});

	return list;
}

void SausageTool::update(float delta_tim) {}
void SausageTool::render_scene() {}
void SausageTool::render_ui() {}