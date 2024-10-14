#pragma once

#include "includes.h"

#include "framework/colors.h"
#include "framework/math/aabb.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include <glm/gtc/quaternion.hpp>

#include <string>

#define MAX_EDITS_PER_EVALUATION 256u
#define MAX_EDITS_PER_SPLINE 255u

enum sdPrimitive : uint32_t {
	SD_SPHERE = 0,
	SD_BOX,
	SD_CAPSULE,
	SD_CONE,
	SD_CYLINDER,
    SD_TORUS,
    SD_VESICA,
	SD_ELLIPSOID,
	SD_PYRAMID,
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

enum ColorBlendOp : uint32_t {
    COLOR_OP_REPLACE = 0,
    COLOR_OP_MIX,
    COLOR_OP_ADDITIVE,
    COLOR_OP_MULTIPLY,
    COLOR_OP_SCREEN,
    /*COLOR_OP_DARKEN,
    COLOR_OP_LIGHTEN,*/
    ALL_COLOR_BLENDING_OPERATIONS
};

struct Edit {
    glm::vec3	position;
    float       dummy0;
    glm::vec4	dimensions = { 0.02f, 0.0f, 0.0f, 0.0f };
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

struct sToUploadStroke {
    uint32_t        stroke_id = 0u;
    uint32_t        edit_count = 0u;
    sdPrimitive     primitive = SD_SPHERE;
    sdOperation     operation = OP_SMOOTH_UNION;//4
    glm::vec4	    parameters = { 0.f, -1.f, 0.f, 0.f }; // 4
    glm::vec3	    aabb_min;// 4
    ColorBlendOp    color_blending_op = ColorBlendOp::COLOR_OP_REPLACE;
    glm::vec3	    aabb_max;
    uint32_t        edit_list_index = 0u;// 4
    // 48 bytes
    StrokeMaterial material;

    AABB get_world_AABB_of_edit_list(const std::vector<Edit> &list) const;
};

class StrokeParameters {

    sdPrimitive     primitive = SD_SPHERE;
    sdOperation     operation = OP_UNION;
    ColorBlendOp    color_blend_op = COLOR_OP_REPLACE;
    glm::vec4       parameters = { 0.f, -1.f, 0.f, 0.005f };
    StrokeMaterial  material = {};

    bool dirty = false;

public:
    void set_primitive(sdPrimitive primitive);
    void set_operation(sdOperation op);
    void set_parameters(const glm::vec4& parameters);
    void set_color_blend_operation(ColorBlendOp op);
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
    ColorBlendOp get_color_blend_operation() const { return color_blend_op; }
    glm::vec4 get_parameters() const { return parameters; }
    StrokeMaterial& get_material() { return material; }
    const StrokeMaterial& get_material() const { return material; }

    bool is_dirty() { return dirty; }
    void set_dirty(bool dirty) { this->dirty = dirty; }
};

struct Stroke {
    uint32_t        stroke_id = 0u;
    uint32_t        edit_count = 0u;
    sdPrimitive     primitive;
    sdOperation     operation;//4
    glm::vec4	    parameters = { 0.f, -1.f, 0.f, 0.f }; // 4
    glm::vec3	    _dummy_;//4
    ColorBlendOp    color_blending_op = ColorBlendOp::COLOR_OP_REPLACE;
    glm::vec4	    _dummy2_;//4
    // 48 bytes
    StrokeMaterial material;

    Edit    edits[MAX_EDITS_PER_EVALUATION] = {};

    AABB get_edit_world_AABB(const uint16_t edit_index) const;
    AABB get_world_AABB() const;
    void get_AABB_intersecting_stroke(const AABB intersection_area, Stroke& resulting_stroke, const uint32_t item_to_exclude = 0u) const;
};

struct sStrokeInfluence {
    uint32_t stroke_count = 0u;
    uint32_t pad_1 = UINT32_MAX; // aligment issues when using vec3
    uint32_t pad_0 = 0u;
    uint32_t pad_2 = UINT32_MAX;
    glm::vec3 eval_aabb_min;
    float pad1;
    glm::vec3 eval_aabb_max;
    float pad2;
    glm::vec4 pad3;
    std::vector<sToUploadStroke> strokes;
};

struct PBRMaterialData {

    Color base_color = colors::BLACK;
    float roughness = 0.0f;
    float metallic = 0.0f;
    glm::vec4 noise_params = glm::vec4(0.0f);
    Color noise_color = colors::RUST;
};

glm::vec3 get_edit_world_half_size(const Edit& edit, const sdPrimitive primitive, const float smooth_margin);
AABB extern_get_edit_world_AABB(const Edit &edit, const sdPrimitive primitive, const float smooth_margin);
