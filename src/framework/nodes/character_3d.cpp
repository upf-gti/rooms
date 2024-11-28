#include "character_3d.h"

#include "framework/nodes/node_factory.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/skeleton_helper_3d.h"
#include "framework/nodes/joint_3d.h"
#include "framework/math/intersections.h"

#include "engine/rooms_engine.h"

#include <fstream>

REGISTER_NODE_CLASS(Character3D)

Character3D::Character3D() : Node3D()
{
    node_type = "Character3D";

    name = "Character";

    set_skeleton(static_cast<RoomsEngine*>(RoomsEngine::instance)->get_default_skeleton());

    helper = new SkeletonHelper3D(skeleton, this);

    // Create structure nodes
    {
        Pose& pose = skeleton->get_rest_pose();
        const auto& indices = skeleton->get_joint_indices();
        const auto& names = skeleton->get_joint_names();
        uint32_t joint_count = indices.size();

        joint_nodes.resize(joint_count);

        for (uint32_t i = 0u; i < joint_count; ++i) {

            SculptNode* new_sculpt = new SculptNode();
            new_sculpt->initialize();
            new_sculpt->set_name(names[i]);
            new_sculpt->set_transform(Transform::combine(get_global_transform(), pose.get_global_transform(i)));
            add_child(new_sculpt);

            Joint3D* joint_3d = new Joint3D();
            joint_3d->set_name(names[i]);
            joint_3d->set_index(i);
            joint_3d->set_pose(&skeleton->get_current_pose());
            joint_3d->set_parent(this);
            joint_3d->set_transform(pose.get_local_transform(i));
            joint_nodes[i] = joint_3d;
        }
    }
}

Character3D::~Character3D()
{
    delete helper;

    const auto& indices = skeleton->get_joint_indices();
    for (uint32_t i = 0u; i < indices.size(); ++i) {
        delete joint_nodes[i];
    }

    // skeleton->unref();
}

void Character3D::update(float delta_time)
{
    if (transform.is_dirty()) {

        Pose& pose = skeleton->get_current_pose();

        // Set pose first
        for (size_t i = 0; i < joint_nodes.size(); ++i) {
            pose.set_local_transform(i, joint_nodes[i]->get_transform());
        }

        // Set transform when all joints in the pose have been updated
        for (size_t i = 0; i < joint_nodes.size(); ++i) {
            Node3D* sculpt = static_cast<Node3D*>(children[i]);
            sculpt->set_transform(Transform::combine(get_global_transform(), pose.get_global_transform(i)));
        }

        transform.set_dirty(false);
    }

    helper->update(delta_time);

    Node3D::update(delta_time);
}

void Character3D::render()
{
    for (auto j : joint_nodes) {
        j->render();
    }

    helper->render();
}

void Character3D::serialize(std::ofstream& binary_scene_file)
{
    Node3D::serialize(binary_scene_file);
}

void Character3D::parse(std::ifstream& binary_scene_file)
{
    Node3D::parse(binary_scene_file);
}

void Character3D::clone(Node* new_node, bool copy)
{
    Node3D::clone(new_node, copy);
}

bool Character3D::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance, Node3D** out)
{
    Pose& pose = skeleton->get_current_pose();

    bool result = false;

    float joint_distance = 1e9f;

    const Transform& global_transform = get_global_transform();

    for (size_t i = 0; i < joint_nodes.size(); ++i) {
        Transform joint_global_transform = Transform::combine(global_transform, pose.get_global_transform(i));
        if (intersection::ray_sphere(ray_origin, ray_direction, joint_global_transform.get_position(), 0.01f, joint_distance)) {
            if (joint_distance < distance) {
                distance = joint_distance;
                result |= true;
                *out = joint_nodes[i];
                (*out)->select();
            }
        }
    }

    return result;
}
