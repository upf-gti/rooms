#include "mirror.h"

#include "utils.h"

void MirrorTool::initialize() {}
void MirrorTool::clean() {}

std::vector<sEdit> MirrorTool::use() {
	std::vector<sEdit> list;

#ifdef XR_SUPPORT
	glm::vec3 edit_origin = Input::get_controller_position(HAND_LEFT);
#else
	glm::vec3 edit_origin = glm::vec3(random_f(), random_f(), random_f());
#endif

	float dist_to_plane = glm::dot(mirror_normal, edit_origin - mirror_origin);

	glm::vec3 rnd_color = glm::vec3(random_f(), random_f(), random_f());

	list.push_back({
			.position = edit_origin,
			.primitive = selected_primitive,
			.color = rnd_color,
			.operation = (is_smooth) ? OP_SMOOTH_UNION : OP_UNION,
			.size = size,
			.radius = radius
		});

	list.push_back({
			.position = edit_origin + mirror_normal * dist_to_plane * 2.0f,
			.primitive = selected_primitive,
			.color = rnd_color,
			.operation = (is_smooth) ? OP_SMOOTH_UNION : OP_UNION,
			.size = size,
			.radius = radius
		});

	return list;
}

void MirrorTool::update(float delta_tim) {}
void MirrorTool::render_scene() {}
void MirrorTool::render_ui() {}