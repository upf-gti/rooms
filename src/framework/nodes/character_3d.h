#pragma once

#include "framework/nodes/node_3d.h"

#include <vector>

class Skeleton;
class Joint3D;
class SkeletonHelper3D;

class Character3D : public Node3D {

    Skeleton* skeleton = nullptr;

    SkeletonHelper3D* helper = nullptr;

    std::vector<Joint3D*> joint_nodes;

    void generate_default_sculpts_skeleton();

public:

    Character3D();
    ~Character3D();

    void initialize();
    void update(float delta_time) override;
    void render() override;
    // void render_gui() override;

    void serialize(std::ofstream& binary_scene_file) override;
    void parse(std::ifstream& binary_scene_file) override;
    void clone(Node* new_node, bool copy = true) override;

    Skeleton* get_skeleton() { return skeleton; }

    void set_skeleton(Skeleton* new_skeleton);

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance, Node3D** out = nullptr) override;
};
