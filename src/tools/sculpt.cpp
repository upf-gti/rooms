#include "sculpt.h"
#include "utils.h"

void SculptTool::initialize() {}
void SculptTool::clean() {}

std::vector<sEdit> SculptTool::use() {
	std::vector<sEdit> list;

	list.push_back(edit_to_add);

	if (use_mirror) {
		float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
		edit_to_add.position = edit_to_add.position + mirror_normal * dist_to_plane * 2.0f;

		list.push_back(edit_to_add);
	}

	return list;
}

void SculptTool::update(float delta_tim) {
#ifdef XR_SUPPORT
	edit_to_add.position = Input::get_controller_position(HAND_LEFT);
#else
	edit_to_add.position = glm::vec3(random_f(), random_f(), random_f());
#endif
}

void SculptTool::render_scene() {}
void SculptTool::render_ui() {}