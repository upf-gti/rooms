#pragma once

#include "framework/nodes/node_3d.h"

#include <vector>

class Group3D : public Node3D {

    static uint32_t last_uid;

public:

    Group3D();

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance) override;
};
