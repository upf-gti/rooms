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
    SD_TORUS,
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
    glm::vec4	dimensions;
    glm::quat   rotation = { 0.f, 0.f, 0.f, 1.f };
    glm::vec4	parameters = { 0.f, -1.f, 0.f, 0.f };

    friend std::ostream& operator<<(std::ostream& os, const Edit& edit);

    std::string to_string() const;
    void parse_string(const std::string& str);

    float weigth_difference(const Edit& edit);

    glm::vec3 world_half_size() const;

    void get_world_AABB(glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation, const bool use_padding = false) const;
};
