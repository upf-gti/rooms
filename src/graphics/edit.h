#pragma once

#include "includes.h"

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
	glm::vec4   rotation;

	friend std::ostream& operator<<(std::ostream& os, const sEdit& edit);

	inline glm::vec3 world_size() const {
		switch (primitive) {
		case SD_SPHERE:
			return glm::vec3(radius, radius, radius) * 2.0f;
		case SD_BOX:
			return size;
		case SD_CAPSULE:
			return glm::abs(position - size) + radius * 2.0f;
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

	inline void get_world_AABB(glm::vec3 *min, glm::vec3 *max, const glm::vec3& start_position) const {
		glm::vec3 h_size = world_size();
		*min = position - start_position - h_size + glm::vec3(0.50f, 0.50f, 0.50f);
		*max = position - start_position + h_size + glm::vec3(0.50f, 0.50f, 0.50f);
	}
};
