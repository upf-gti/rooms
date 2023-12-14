#include "edit.h"
#include <sstream>
#include "spdlog/spdlog.h"

std::ostream& operator<<(std::ostream& os, const Edit& edit)
{
    os << "Position: " << edit.position.x << ", " << edit.position.y << ", " << edit.position.z << std::endl;
    //os << "Primitive: " << edit.primitive << std::endl;
    //os << "Operation: " << edit.operation << std::endl;
    os << "Dimensions: " << edit.dimensions.x << ", " << edit.dimensions.y << ", " << edit.dimensions.z << edit.dimensions.w << std::endl;
    return os;
}

std::string Edit::to_string() const {

    std::string text;
    text += std::to_string(position.x) + " " + std::to_string(position.y) + " " + std::to_string(position.z) + "/";
    //text += std::to_string(primitive) + "/";
    //text += std::to_string(operation) + "/";
    text += std::to_string(dimensions.x) + " " + std::to_string(dimensions.y) + " " + std::to_string(dimensions.z) + " " + std::to_string(dimensions.w) + "/";
    text += std::to_string(rotation.x) + " " + std::to_string(rotation.y) + " " + std::to_string(rotation.z) + " " + std::to_string(rotation.w) + "/";
    //text += std::to_string(parameters.x) + " " + std::to_string(parameters.y) + " " + std::to_string(parameters.z) + " " + std::to_string(parameters.w);
    return text;
}

void Edit::parse_string(const std::string& str) {

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
    //primitive   = (sdPrimitive)std::atoi(tokens[1].c_str());
    //operation   = (sdOperation)std::atoi(tokens[3].c_str());
    dimensions  = load_vec4(tokens[4]);
    rotation    = load_quat(tokens[5]);
    //parameters  = load_vec4(tokens[6]);
}

float Edit::weigth_difference(const Edit& edit) {

    float position_diff = glm::length(position - edit.position);
    glm::quat quat_diff = rotation * glm::inverse(edit.rotation);
    float angle_diff = 2.0f * atan2(glm::length(glm::vec3(quat_diff.x, quat_diff.y, quat_diff.z)), quat_diff.w);
    float size_diff = glm::length(dimensions - edit.dimensions);

    return position_diff + angle_diff + size_diff;
}

void StrokeParameters::set_operation(sdOperation op)
{
    operation = op;
    was_operation_changed = true;
}

bool StrokeParameters::must_change_stroke(const StrokeParameters& p)
{
    bool result = false;

    result |= (primitive != p.primitive);
    result |= (parameters != p.parameters);
    result |= (color != p.color);
    result |= (material != p.material);
    result |= was_operation_changed;

    return result;
}

glm::vec3 Stroke::get_edit_world_half_size(const uint8_t edit_index) const {

    glm::vec3 size = glm::vec3(edits[edit_index].dimensions);
    float radius = edits[edit_index].dimensions.w;

    switch (primitive) {
    case SD_SPHERE:
        return glm::vec3(size.x);
    case SD_BOX:
        return size;
    case SD_CAPSULE:
        return glm::abs(edits[edit_index].position - size) + radius;
    case SD_CONE:
     	return glm::abs(edits[edit_index].position - size) + radius * 2.0f;
        //case SD_PYRAMID:
        //	return glm::abs(position - size) + radius * 2.0f;
    case SD_CYLINDER:
        return glm::abs(edits[edit_index].position - size) + radius * 2.0f;
    case SD_TORUS:
        return glm::abs(size) + radius * 2.0f;
    default:
        assert(false);
        return {};
    }
}

void Stroke::get_edit_world_AABB(const uint8_t edit_index, glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation) const {
    glm::vec3 pure_edit_half_size = get_edit_world_half_size(edit_index);

    // TODO: Add smooth margin

    glm::vec3 rotated_mx_size = glm::vec3(-1000.0f, -1000.0f, -1000.0f);
    glm::vec3 rotated_min_size = glm::vec3(1000.0f, 1000.0f, 1000.0f);

    glm::quat edit_rotation = { 0.0, 0.0, 0.0, 1.0 };

    const Edit& edit = edits[edit_index];

    if (primitive != SD_SPHERE) {
        edit_rotation = edits[edit_index].rotation;
    }

    // Rotate the AABB (turning it into an OBB) and compute the AABB
    const glm::vec3 axis[8] = { edit_rotation * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x,  pure_edit_half_size.y, -pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y,  pure_edit_half_size.z),
                                edit_rotation * glm::vec3(-pure_edit_half_size.x, -pure_edit_half_size.y, -pure_edit_half_size.z) };

    for (uint8_t i = 0; i < 8; i++) {
        rotated_mx_size.x = glm::max(rotated_mx_size.x, axis[i].x);
        rotated_mx_size.y = glm::max(rotated_mx_size.y, axis[i].y);
        rotated_mx_size.z = glm::max(rotated_mx_size.z, axis[i].z);

        rotated_min_size.x = glm::min(rotated_min_size.x, axis[i].x);
        rotated_min_size.y = glm::min(rotated_min_size.y, axis[i].y);
        rotated_min_size.z = glm::min(rotated_min_size.z, axis[i].z);
    }

    const glm::vec3 edit_half_size = (rotated_mx_size - rotated_min_size) / 2.0f;

    *min = (sculpt_rotation * (edits[edit_index].position) - edit_half_size);
    *max = (sculpt_rotation * (edits[edit_index].position) + edit_half_size);
}


void Stroke::get_world_AABB(glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation) const {
    glm::vec3 it_min = glm::vec3(FLT_MAX), it_max = glm::vec3(-FLT_MAX);
    glm::vec3 edit_min, edit_max;
    for (uint8_t i = 0u; i < edit_count; i++) {
        get_edit_world_AABB(i, &edit_min, &edit_max, start_position, sculpt_rotation);

        it_min = glm::min(edit_min, it_min);
        it_max = glm::max(edit_max, it_max);
    }

    *min = it_min;
    *max = it_max;
}
