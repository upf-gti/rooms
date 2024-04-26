#include "edit.h"

#include "framework/utils/intersections.h"
#include "framework/utils/math_utils.h"

#include <cfloat>
#include <sstream>
#include <vector>

std::ostream& operator<<(std::ostream& os, const Edit& edit)
{
    os << "Position: " << edit.position.x << ", " << edit.position.y << ", " << edit.position.z << std::endl;
    //os << "Primitive: " << edit.primitive << std::endl;
    //os << "Operation: " << edit.operation << std::endl;
    os << "Dimensions: " << edit.dimensions.x << ", " << edit.dimensions.y << ", " << edit.dimensions.z << edit.dimensions.w << std::endl;
    return os;
}

std::string Edit::to_string() const
{
    std::string text;
    text += std::to_string(position.x) + " " + std::to_string(position.y) + " " + std::to_string(position.z) + "/";
    text += std::to_string(dimensions.x) + " " + std::to_string(dimensions.y) + " " + std::to_string(dimensions.z) + " " + std::to_string(dimensions.w) + "/";
    text += std::to_string(rotation.x) + " " + std::to_string(rotation.y) + " " + std::to_string(rotation.z) + " " + std::to_string(rotation.w) + "/";
    return text;
}

void Edit::parse_string(const std::string& str)
{
    size_t pos = 0;
    std::vector<std::string> tokens;
    std::string s = str;

    while ((pos = s.find("/")) != std::string::npos) {
        tokens.push_back(s.substr(0, pos));
        s = s.substr(pos + 1);
    }

    // Add last token...
    tokens.push_back(s.substr(0, pos));

    // Set data...
    position    = load_vec3(tokens[0]);
    dimensions  = load_vec4(tokens[1]);
    rotation    = load_quat(tokens[2]);
}

float Edit::weigth_difference(const Edit& edit)
{
    float position_diff = glm::length(position - edit.position);
    glm::quat quat_diff = rotation * glm::inverse(edit.rotation);
    float angle_diff = 2.0f * atan2(glm::length(glm::vec3(quat_diff.x, quat_diff.y, quat_diff.z)), quat_diff.w);
    float size_diff = glm::length(dimensions - edit.dimensions);

    return position_diff + angle_diff + size_diff;
}

void StrokeParameters::set_primitive(sdPrimitive primitive)
{
    this->primitive = primitive;
    dirty = true;
}

void StrokeParameters::set_operation(sdOperation operation)
{
    this->operation = operation;
    dirty = true;
}

void StrokeParameters::set_parameters(const glm::vec4& parameters)
{
    this->parameters = parameters;
    dirty = true;
}

void StrokeParameters::set_color_blend_operation(ColorBlendingOp op)
{
    this->color_blend_op = op;
    dirty = true;
}

void StrokeParameters::set_smooth_factor(const float smooth_factor)
{
    parameters.w = smooth_factor;
    dirty = true;
}

void StrokeParameters::set_material(const StrokeMaterial& material)
{
    this->material = material;
    dirty = true;
}

void StrokeParameters::set_material_color(const Color& color)
{
    material.color = color;
    dirty = true;
}

void StrokeParameters::set_material_roughness(float roughness)
{
    material.roughness = roughness;
    dirty = true;
}

void StrokeParameters::set_material_metallic(float metallic)
{
    material.metallic = metallic;
    dirty = true;
}

void StrokeParameters::set_material_noise(float intensity, float frequency, int octaves)
{
    material.noise_params.x = intensity != -1.0f ? intensity : material.noise_params.x;
    material.noise_params.y = frequency != -1.0f ? frequency : material.noise_params.y;
    material.noise_params.z = octaves   != -1    ? octaves   : material.noise_params.z;
    dirty = true;
}

void StrokeParameters::set_material_noise_color(const Color& color)
{
    material.noise_color = color;
    dirty = true;
}

glm::vec3 Stroke::get_edit_world_half_size(const uint8_t edit_index) const
{
    glm::vec3 size = glm::vec3(edits[edit_index].dimensions);
    float size_param = edits[edit_index].dimensions.w;
    float radius = edits[edit_index].dimensions.x;

    const glm::vec3 smooth_margin = (operation == OP_SMOOTH_PAINT || operation == OP_SMOOTH_UNION || operation == OP_SMOOTH_SUBSTRACTION) ? glm::vec3(parameters.w * 2.0f) : glm::vec3(0.0f);

    switch (primitive) {
    case SD_SPHERE:
        return glm::vec3(size.x) + smooth_margin;
    case SD_BOX:
        return size + smooth_margin;
    case SD_CAPSULE:
        return glm::vec3(size_param) + glm::vec3(0.0f, radius / 2.0f, 0.0f);// +smooth_margin;
    case SD_CONE:
        return glm::vec3(size_param, radius, size_param) + smooth_margin;
    //case SD_PYRAMID:
        //	return glm::abs(position - size) + radius * 2.0f;
    case SD_CYLINDER:
        return glm::vec3(size_param, radius, size_param) + smooth_margin;
    case SD_TORUS:
        return glm::abs(size) + radius * 2.0f + smooth_margin;
    /*case SD_BEZIER:
        return glm::vec3(0.0f);*/
    default:
        assert(false);
        return {};
    }
}

AABB Stroke::get_edit_world_AABB(const uint8_t edit_index) const
{
    // Special case for the capsule
    if (primitive == SD_CAPSULE) {
        const float radius = edits[edit_index].dimensions.w;
        const float height = edits[edit_index].dimensions.x;
        const float smooht_margin = parameters.w;

        AABB a1 = { edits[edit_index].position, glm::vec3(radius)};
        AABB a2 = { edits[edit_index].position + (glm::inverse(edits[edit_index].rotation) * glm::vec3(0.0f, height, 0.0f)), glm::vec3(radius) };

        return merge_aabbs(a1, a2);
    }

    glm::vec3 pure_edit_half_size = get_edit_world_half_size(edit_index);

    glm::quat edit_rotation = { 0.0, 0.0, 0.0, 1.0 };

    const Edit& edit = edits[edit_index];

    if (primitive != SD_SPHERE) {
        edit_rotation = edits[edit_index].rotation;
    }

    glm::vec3 aabb_center = edits[edit_index].position;

    // Rotate the AABB (turning it into an OBB) and compute the AABB
    const glm::vec3 axis[8] = { edit_rotation * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z) };


    glm::vec3 rotated_max_size = glm::vec3(-FLT_MAX);
    glm::vec3 rotated_min_size = glm::vec3(FLT_MAX);

    for (uint8_t i = 0; i < 8; i++) {
        rotated_max_size.x = glm::max(rotated_max_size.x, axis[i].x);
        rotated_max_size.y = glm::max(rotated_max_size.y, axis[i].y);
        rotated_max_size.z = glm::max(rotated_max_size.z, axis[i].z);

        rotated_min_size.x = glm::min(rotated_min_size.x, axis[i].x);
        rotated_min_size.y = glm::min(rotated_min_size.y, axis[i].y);
        rotated_min_size.z = glm::min(rotated_min_size.z, axis[i].z);
    }

    const glm::vec3 edit_half_size = (rotated_max_size - rotated_min_size);

    return { aabb_center, edit_half_size };
}


AABB Stroke::get_world_AABB() const
{
    AABB world_aabb;
    for (uint8_t i = 0u; i < edit_count; i++) {
        AABB aabb = get_edit_world_AABB(i);
        world_aabb = merge_aabbs(world_aabb, aabb);
    }

    return world_aabb;
}

// Compute the resulting stroke on the current stroke's edits that are inside the area
void Stroke::get_AABB_intersecting_stroke(const AABB intersection_area,
                                                Stroke& resulting_stroke) const {
    resulting_stroke.edit_count = 0u;
    resulting_stroke.primitive = primitive;
    resulting_stroke.operation = operation;
    resulting_stroke.color_blending_op = color_blending_op;
    resulting_stroke.parameters = parameters;
    resulting_stroke.material = material;
    resulting_stroke.stroke_id = stroke_id;

    for (uint16_t i = 0u; i < edit_count; i++) {
        if (intersection::AABB_AABB_min_max(intersection_area, get_edit_world_AABB(i))) {
            resulting_stroke.edits[resulting_stroke.edit_count++] = edits[i];
        }
    }
}
