#pragma once

#include "framework/nodes/node_3d.h"

#include <vector>

class Group3D : public Node3D {

    static uint32_t last_uid;

    void update_pivot();

public:

    Group3D();

    void add_node(Node3D* node);
    void add_nodes(const std::vector<Node3D*>& nodes);

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance) override;
};
