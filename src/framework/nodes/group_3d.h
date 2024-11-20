#pragma once

#include "framework/nodes/node_3d.h"

#include <vector>

class Group3D : public Node3D {

    static uint32_t last_uid;

public:

    Group3D();

    void serialize(std::ofstream& binary_scene_file) override;
    void parse(std::ifstream& binary_scene_file) override;

    void clone(Node* new_node, bool copy = true) override;

    void add_node(Node3D* node, bool pivot_dirty = true);
    void add_nodes(const std::vector<Node3D*>& nodes, bool pivot_dirty = true);

    void update_pivot();

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance, Node3D** out = nullptr) override;
};
