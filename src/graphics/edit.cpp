#include "edit.h"

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
        return size + glm::vec3(radius);
    case SD_CAPSULE:
        return glm::abs(position - size) + radius;
        //case SD_CONE:
        //	return glm::abs(position - size) + radius * 2.0f;
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

    // Avoid rotation if primitive is a sphere, and the rotation is unitary
    //if (primitive == SD_SPHERE || glm::all(glm::epsilonEqual(rotation, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0.001f))) {
    //	rotated_mx_size = pure_edit_half_size;
    //	rotated_min_size = -pure_edit_half_size;
    //} else {

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
    //}

    const glm::vec3 edit_half_size = (rotated_mx_size - rotated_min_size) / 2.0f;

    *min = (sculpt_rotation * (position - start_position) - edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
    *max = (sculpt_rotation * (position - start_position) + edit_half_size + glm::vec3(0.50f, 0.50f, 0.50f));
}
