#pragma once

#include "includes.h"

enum sdPrimitive {
	SD_SPHERE = 0,
	SD_BOX,
	SD_ELLIPSOID,
	SD_CONE,
	SD_PYRAMID,
	SD_CYLINDER,
	SD_CAPSULE,
	SD_TORUS,
	SD_CAPPED_TORUS,
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

struct Edit {

	glm::vec3	position;
	sdPrimitive primitive;
	glm::vec3	color;
	sdOperation operation;
	glm::vec3	size;
	float		radius;

};