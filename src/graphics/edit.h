#pragma once

#include "includes.h"

#include "framework/colors.h"
#include "framework/aabb.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include <glm/gtc/quaternion.hpp>

#include <string>

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

enum ColorBlendingOp : uint32_t {
    COLOR_OP_REPLACE = 0,
    COLOR_OP_MIX,
    COLOR_OP_ADDITIVE,
    COLOR_OP_MULTIPLY,
    COLOR_OP_SCREEN,
    COLOR_OP_DARKEN,
    COLOR_OP_LIGHTEN,
    ALL_COLOR_BLENDING_OPERATIONS
};

struct Edit {
    glm::vec3	position;
    float       dummy0;
    glm::vec4	dimensions = { 0.3f, 0.05f, 0.05f, 0.05f };
    glm::quat   rotation = { 0.f, 0.f, 0.f, 1.f };
    glm::vec4   padding;

    std::string to_string() const;
    void parse_string(const std::string& str);

    float weigth_difference(const Edit& edit);
};

struct StrokeMaterial {

    float roughness = 0.7f;
    float metallic  = 0.2f;
    float emissive  = 0.2f;
    float dummy0    = 0.0f;

    glm::vec4	color           = colors::RED;
    glm::vec4   noise_params    = glm::vec4(0.0f, 20.0f, 8.0f, 1.0f); // intensity, frequency, octaves, unused
    Color       noise_color     = colors::WHITE;
};

class StrokeParameters {

    sdPrimitive     primitive = SD_SPHERE;
    sdOperation     operation = OP_UNION;
    ColorBlendingOp color_blend_op = COLOR_OP_REPLACE;
    glm::vec4       parameters = { 0.f, -1.f, 0.f, 0.01f };
    StrokeMaterial  material = {};

    bool dirty = false;

public:
    void set_primitive(sdPrimitive primitive);
    void set_operation(sdOperation op);
    void set_parameters(const glm::vec4& parameters);
    void set_color_blend_operation(ColorBlendingOp op);
    void set_smooth_factor(const float smooth_factor);
    void set_material(const StrokeMaterial& material);
    void set_material_color(const Color& color);
    void set_material_roughness(float roughness);
    void set_material_metallic(float metallic);
    void set_material_noise(float intensity = -1.0f, float frequency = -1.0f, int octaves = -1);
    void set_material_noise_color(const Color& color);

    float get_smooth_factor() const { return parameters.w; }
    sdPrimitive get_primitive() const { return primitive; }
    sdOperation get_operation() const { return operation; }
    ColorBlendingOp get_color_blending_operation() const { return color_blend_op; }
    glm::vec4 get_parameters() const { return parameters; }
    StrokeMaterial& get_material() { return material; }
    const StrokeMaterial& get_material() const { return material; }

    bool is_dirty() { return dirty; }
    void set_dirty(bool dirty) { this->dirty = dirty; }
};

struct Stroke {
    uint32_t    stroke_id;
    uint32_t    edit_count = 0u;
    sdPrimitive primitive;
    sdOperation operation;
    glm::vec4	parameters = { 0.f, -1.f, 0.f, 0.f };
    glm::vec3	_dummy_;
    ColorBlendingOp color_blending_op = ColorBlendingOp::COLOR_OP_REPLACE;
    glm::vec4	_dummy2_;

    // 48 bytes
    StrokeMaterial material;

    Edit    edits[MAX_EDITS_PER_EVALUATION] = {};

    glm::vec3 get_edit_world_half_size(const uint8_t edit_index) const;
    AABB get_edit_world_AABB(const uint8_t edit_index) const;
    AABB get_world_AABB() const;
    void get_AABB_intersecting_stroke(const AABB intersection_area, Stroke& resulting_stroke) const;
};

struct PBRMaterialData {

    Color base_color = colors::BLACK;
    float roughness = 0.0f;
    float metallic = 0.0f;
    glm::vec4 noise_params = glm::vec4(0.0f);
    Color noise_color = colors::RUST;
};
