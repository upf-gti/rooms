#pragma once

#include "framework/nodes/skeleton_instance_3d.h"

class Character3D : public SkeletonInstance3D {

    void generate_default_sculpts_skeleton();

public:

    Character3D();
    ~Character3D();

    void initialize();
    void update(float delta_time) override;
    void render() override;
    void render_gui() override;

    void serialize(std::ofstream& binary_scene_file) override;
    void parse(std::ifstream& binary_scene_file) override;
    void clone(Node* new_node, bool copy = true) override;

    void set_skeleton(Skeleton* new_skeleton);

    void update_joints_from_pose() override;
};
