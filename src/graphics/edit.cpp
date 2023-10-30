#include "edit.h"
#include <sstream>

std::ostream& operator<<(std::ostream& os, const Edit& edit)
{
    os << "Position: " << edit.position.x << ", " << edit.position.y << ", " << edit.position.z << std::endl;
    os << "Primitive: " << edit.primitive << std::endl;
    os << "Color: " << edit.color.x << ", " << edit.color.y << ", " << edit.color.z << std::endl;
    os << "Operation: " << edit.operation << std::endl;
    os << "Dimensions: " << edit.dimensions.x << ", " << edit.dimensions.y << ", " << edit.dimensions.z << edit.dimensions.w << std::endl;
    return os;
}

std::string Edit::to_string() const {

    std::string text;
    text += std::to_string(position.x) + " " + std::to_string(position.y) + " " + std::to_string(position.z) + "/";
    text += std::to_string(primitive) + "/";
    text += std::to_string(color.x) + " " + std::to_string(color.y) + " " + std::to_string(color.z) + "/";
    text += std::to_string(operation) + "/";
    text += std::to_string(dimensions.x) + " " + std::to_string(dimensions.y) + " " + std::to_string(dimensions.z) + " " + std::to_string(dimensions.w) + "/";
    text += std::to_string(rotation.x) + " " + std::to_string(rotation.y) + " " + std::to_string(rotation.z) + " " + std::to_string(rotation.w) + "/";
    text += std::to_string(parameters.x) + " " + std::to_string(parameters.y) + " " + std::to_string(parameters.z) + " " + std::to_string(parameters.w);
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
    primitive   = (sdPrimitive)std::atoi(tokens[1].c_str());
    color       = load_vec3(tokens[2]);
    operation   = (sdOperation)std::atoi(tokens[3].c_str());
    dimensions  = load_vec4(tokens[4]);
    rotation    = load_quat(tokens[5]);
    parameters  = load_vec4(tokens[6]);
}

float Edit::weigth_difference(const Edit& edit) {

    float position_diff = glm::length(position - edit.position);
    glm::quat quat_diff = rotation * glm::inverse(edit.rotation);
    float angle_diff = 2.0f * atan2(glm::length(glm::vec3(quat_diff.x, quat_diff.y, quat_diff.z)), quat_diff.w);
    float size_diff = glm::length(dimensions - edit.dimensions);

    return position_diff + angle_diff + size_diff;
}

glm::vec3 Edit::world_half_size() const {

    glm::vec3 size = glm::vec3(dimensions);
    float radius = dimensions.w;

    switch (primitive) {
    case SD_SPHERE:
        return size;
    case SD_BOX:
        return size;
    case SD_CAPSULE:
        return glm::abs(position - size) + radius;
    case SD_CONE:
     	return glm::abs(position - size) + radius * 2.0f;
        //case SD_PYRAMID:
        //	return glm::abs(position - size) + radius * 2.0f;
    case SD_CYLINDER:
        return glm::abs(position - size) + radius * 2.0f;
    case SD_TORUS:
        return glm::abs(size) + radius * 2.0f;
    default:
        assert(false);
        return {};
    }
}

void Edit::get_world_AABB(glm::vec3* min, glm::vec3* max, const glm::vec3& start_position, const glm::quat& sculpt_rotation, const bool use_padding) const {
    glm::vec3 pure_edit_half_size = world_half_size();

    if (use_padding) {
        pure_edit_half_size += 0.04f;
    }

    glm::vec3 rotated_mx_size = glm::vec3(-1000.0f, -1000.0f, -1000.0f);
    glm::vec3 rotated_min_size = glm::vec3(1000.0f, 1000.0f, 1000.0f);

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

    const glm::vec3 edit_half_size = (rotated_mx_size - rotated_min_size) / 2.0f;

    *min = (sculpt_rotation * (position - start_position) - edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
    *max = (sculpt_rotation * (position - start_position) + edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
}
