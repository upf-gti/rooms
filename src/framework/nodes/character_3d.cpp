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

void Character3D::initialize()
{
    set_skeleton(static_cast<RoomsEngine*>(RoomsEngine::instance)->get_default_skeleton());

    helper = new SkeletonHelper3D(skeleton, this);

    generate_default_sculpts_skeleton();
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

    // Serialize skeleton..
    assert(skeleton);
    skeleton->serialize(binary_scene_file);
}

void Character3D::parse(std::ifstream& binary_scene_file)
{
    Node3D::parse(binary_scene_file);

    skeleton = new Skeleton();
    skeleton->parse(binary_scene_file);

    set_skeleton(skeleton);

    helper = new SkeletonHelper3D(skeleton, this);
}

void Character3D::clone(Node* new_node, bool copy)
{
    Node3D::clone(new_node, copy);
}

void Character3D::set_skeleton(Skeleton* new_skeleton)
{
    skeleton = new_skeleton;

    uint32_t joint_count = skeleton->get_joints_count();

    if (joint_count == 0u) {
        assert("Character must have a skeleton with joints!");
        return;
    }

    // Create structure nodes and fill animatable properties
    // from the entire skeleton
    Pose& pose = skeleton->get_rest_pose();
    const auto& names = skeleton->get_joint_names();

    joint_nodes.resize(joint_count);

    for (uint32_t i = 0u; i < joint_count; ++i) {
        Joint3D* joint_3d = new Joint3D();
        joint_3d->set_name(names[i]);
        joint_3d->set_index(i);
        joint_3d->set_pose(&skeleton->get_current_pose());
        joint_3d->set_parent(this);
        joint_3d->set_transform(pose.get_local_transform(i));
        joint_nodes[i] = joint_3d;

        animatable_properties[joint_3d->get_name() + "$translation"] = { AnimatablePropertyType::FVEC3, &joint_3d->get_transform().get_position_ref() };
        animatable_properties[joint_3d->get_name() + "$rotation"] = { AnimatablePropertyType::QUAT, &joint_3d->get_transform().get_rotation_ref() };
        animatable_properties[joint_3d->get_name() + "$scale"] = { AnimatablePropertyType::FVEC3, &joint_3d->get_transform().get_scale_ref() };
    }
}

void Character3D::generate_default_sculpts_skeleton()
{
    uint32_t joint_count = skeleton->get_joints_count();

    if (joint_count == 0u) {
        assert("Character must have a skeleton with joints!");
        return;
    }

    Pose& pose = skeleton->get_rest_pose();
    const auto& names = skeleton->get_joint_names();

    for (uint32_t i = 0u; i < joint_count; ++i) {
        SculptNode* new_sculpt = new SculptNode();
        new_sculpt->initialize();
        new_sculpt->set_name(names[i]);
        new_sculpt->set_transform(Transform::combine(get_global_transform(), pose.get_global_transform(i)));
        new_sculpt->set_from_memory(true);
        add_child(new_sculpt);
    }
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
                Joint3D::selected_joint = static_cast<Joint3D*>(*out);
            }
        }
    }

    return result;
}