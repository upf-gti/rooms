#pragma once

#include "includes.h"
#include "utils.h"

#define MAX_EDITS_PER_EVALUATION 64

enum sdPrimitive : uint32_t {
	SD_SPHERE = 0,
	SD_BOX,
	SD_ELLIPSOID,
	SD_CONE,
	SD_PYRAMID,
	SD_CYLINDER,
	SD_CAPSULE,
    SD_TORUS,
    SD_BEZIER,
	ALL_PRIMITIVES
};

enum sdOperation : uint32_t {
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

struct alignas(16) Edit {
    glm::vec3	position;
    glm::vec4	dimensions;
    glm::quat   rotation = { 0.f, 0.f, 0.f, 1.f };

    std::string to_string() const;
    void parse_string(const std::string& str);

    float weigth_difference(const Edit& edit);
};

struct StrokeParameters {
    sdPrimitive primitive = SD_SPHERE;
    sdOperation operation = OP_UNION;
    glm::vec4   parameters = { 0.f, -1.f, 0.f, 0.f };
    glm::vec4   color = { 0.f, 0.f, 0.f, 0.f };

    bool was_operation_changed = false;

    bool must_change_stroke(const StrokeParameters& p);
};

struct alignas(256) Stroke {
    uint32_t    stroke_id;
    uint32_t    edit_count = 0u;
    sdPrimitive primitive;
    sdOperation operation;
    glm::vec4	parameters = { 0.f, -1.f, 0.f, 0.f };
    glm::vec4	color;

    Edit        edits[MAX_EDITS_PER_EVALUATION] = {};

    glm::vec3 get_edit_world_half_size(const uint8_t edit_index) const;
    void get_edit_world_AABB(const uint8_t edit_index, glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation) const;
    void get_world_AABB(glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation) const;
    StrokeParameters as_params() { return { primitive, operation, parameters, color }; }
};
