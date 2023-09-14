#pragma once

#include "includes.h"
#include "utils.h"

#include <iostream>

enum sdPrimitive {
	SD_SPHERE = 0,
	SD_BOX,
	SD_ELLIPSOID,
	SD_CONE,
	SD_PYRAMID,
	SD_CYLINDER,
	SD_CAPSULE,
	ALL_PRIMITIVES
};

enum sdOperation {
	OP_UNION = 0,
	OP_SUBSTRACTION,
	OP_INTERSECTION,
	OP_PAINT,
	OP_SMOOTH_UNION,
	OP_SMOOTH_SUBSTRACTION,
	OP_SMOOTH_INTERSECTION,
	OP_SMOOTH_PAINT,
	ALL_OPERATIONS
};

struct sEdit {
	glm::vec3	position;
	sdPrimitive primitive;
	glm::vec3	color;
	sdOperation operation;
	glm::vec3	size = {};
	float		radius = 1.0f;
	glm::quat   rotation;

	friend std::ostream& operator<<(std::ostream& os, const sEdit& edit);

	inline glm::vec3 world_half_size() const {
		switch (primitive) {
		case SD_SPHERE:
			return glm::vec3(radius, radius, radius);
		case SD_BOX:
			return size + glm::vec3(radius, radius, radius);
		case SD_CAPSULE:
			return glm::abs(position - size) + radius;
		//case SD_CONE:
		//	return glm::abs(position - size) + radius * 2.0f;
		//case SD_PYRAMID:
		//	return glm::abs(position - size) + radius * 2.0f;
		//case SD_CYLINDER:
		//	return glm::abs(position - size) + radius * 2.0f;
		default:
			assert(false);
			return {};
		}
	}

	inline void get_world_AABB(glm::vec3 *min, glm::vec3 *max, const glm::vec3 & start_position, const glm::quat& sculpt_rotation, const bool use_padding = false) const {
		glm::vec3 pure_edit_half_size = world_half_size();
		
		if (use_padding) {
			pure_edit_half_size += 0.04f;
		}

		glm::vec3 rotated_mx_size = glm::vec3(-1000.0f, -1000.0f, -1000.0f);
		glm::vec3 rotated_min_size = glm::vec3(1000.0f, 1000.0f, 1000.0f);

		// Avoid rotation if primitive is a sphere, and the rotation is unitary
		//if (primitive == SD_SPHERE || glm::all(glm::epsilonEqual(rotation, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0.001f))) {
		//	rotated_mx_size = pure_edit_half_size;
		//	rotated_min_size = -pure_edit_half_size;
		//} else {

			glm::quat quat_rot = glm::quat{ rotation.w, rotation.x, rotation.y, rotation.z };

			// Rotate the AABB (turning it into an OBB) and compute the AABB
			const glm::vec3 axis[8] = { quat_rot * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
										quat_rot * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
										quat_rot * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
										quat_rot * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z),
										quat_rot * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
										quat_rot * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
										quat_rot * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
										quat_rot * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z) };

			for (uint8_t i = 0; i < 8; i++) {
				rotated_mx_size.x = glm::max(rotated_mx_size.x, axis[i].x);
				rotated_mx_size.y = glm::max(rotated_mx_size.y, axis[i].y);
				rotated_mx_size.z = glm::max(rotated_mx_size.z, axis[i].z);

				rotated_min_size.x = glm::min(rotated_min_size.x, axis[i].x);
				rotated_min_size.y = glm::min(rotated_min_size.y, axis[i].y);
				rotated_min_size.z = glm::min(rotated_min_size.z, axis[i].z);
			}
		//}

		const glm::vec3 edit_half_size = (rotated_mx_size - rotated_min_size) / 2.0f;

		*min = (sculpt_rotation * (position - start_position) - edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
		*max = (sculpt_rotation * (position - start_position) + edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
	}
};
