#include "group_3d.h"

#include "framework/math/intersections.h"

uint32_t Group3D::last_uid = 0;

Group3D::Group3D()
{
    node_type = "Group3D";
    name = "Group" + std::to_string(last_uid++);
}

void Group3D::update_pivot()
{
    size_t child_count = get_children().size();

    if (child_count == 0u) {
        assert(0 && "Should not happen..");
        return;
    }

    // Set pivot at new position and transform
    // models of the childs to keep its world position

    auto& children = get_children();

    if (child_count == 1u) {
        Node3D* child = static_cast<Node3D*>(children[0]);
        set_position(child->get_translation());
        child->set_position({ 0.0f, 0.0f, 0.0f });
    }
    else {
        glm::vec3 average_position = { 0.0f, 0.0f, 0.0f };
        for (Node* child : children) {
            Node3D* child_3d = static_cast<Node3D*>(child);
            average_position += child_3d->get_translation();
        }

        average_position /= (float)children.size();
        glm::vec3 offset = average_position - get_translation();
        translate(offset);

        for (Node* child : children) {
            Node3D* child_3d = static_cast<Node3D*>(child);
            //Transform t = Transform::mat4_to_transform(glm::inverse(get_global_model()) * child_3d->get_model());
            child_3d->translate(-offset);
        }
    }
}

void Group3D::add_node(Node3D* node, bool pivot_dirty)
{
    if (children.empty()) {
        set_position(static_cast<Node3D*>(node)->get_translation());
    }

    add_child(node);

    node->set_position(node->get_local_translation() - get_translation());

    if (pivot_dirty) {
        update_pivot();
    }
}

void Group3D::add_nodes(const std::vector<Node3D*>& nodes, bool pivot_dirty)
{
    for (auto& node : nodes) {
        add_node(node, false);
    }

    if (pivot_dirty) {
        update_pivot();
    }
}

bool Group3D::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance)
{
    for (auto node : get_children()) {

        Node3D* node_3d = dynamic_cast<Node3D*>(node);

        if (!node_3d) {
            continue;
        }

        if (node_3d->test_ray_collision(ray_origin, ray_direction, distance)) {
            return true;
        }
    }

    return false;
}
