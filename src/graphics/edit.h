#pragma once

#include "includes.h"
#include "utils.h"
#include "framework/colors.h"
#include "framework/aabb.h"

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
    float       dummy0;
    glm::vec4	dimensions;
    glm::quat   rotation = { 0.f, 0.f, 0.f, 1.f };

    std::string to_string() const;
    void parse_string(const std::string& str);

    float weigth_difference(const Edit& edit);
};

class StrokeParameters {
    sdPrimitive primitive = SD_SPHERE;
    sdOperation operation = OP_UNION;
    glm::vec4   parameters = { 0.f, -1.f, 0.f, 0.0f };
    Color       color = colors::RED;
    glm::vec4   material = { 0.7f, 0.2f, 0.f, 0.f }; // rough, metallic, emissive, unused

    bool dirty = false;

public:
    void set_primitive(sdPrimitive primitive);
    void set_operation(sdOperation op);
    void set_parameters(const glm::vec4& parameters);
    void set_smooth_factor(const float smooth_factor);
    void set_color(const Color& color);
    void set_material(const glm::vec4& material);
    void set_material_roughness(float roughness);
    void set_material_metallic(float metallic);

    sdPrimitive get_primitive() const { return primitive; }
    sdOperation get_operation() const { return operation; }
    glm::vec4 get_parameters() const { return parameters; }
    Color get_color() const  { return color; }
    glm::vec4 get_material() const { return material; }

    bool is_dirty() { return dirty; }
    void set_dirty(bool dirty) { this->dirty = dirty; }

};

struct alignas(256) Stroke {
    uint32_t    stroke_id;
    uint32_t    edit_count = 0u;
    sdPrimitive primitive;
    sdOperation operation;
    glm::vec4	parameters = { 0.f, -1.f, 0.f, 0.f };
    glm::vec4	color;
    glm::vec4   material = { 0.7f, 0.2f, 0.f, 0.f }; // rough, metallic, emissive, unused

    Edit        edits[MAX_EDITS_PER_EVALUATION] = {};

    glm::vec3 get_edit_world_half_size(const uint8_t edit_index) const;
    AABB get_edit_world_AABB(const uint8_t edit_index) const;
    AABB get_world_AABB() const;
};
