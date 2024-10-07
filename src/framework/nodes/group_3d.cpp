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

    //debug, use always pos of the first node
    set_position(static_cast<Node3D*>(children[0])->get_translation());

    /*if (child_count == 1u) {
        set_position(static_cast<Node3D*>(children[0])->get_translation());
    }
    else {

    }*/

    for (Node* child : children) {
        Node3D* child_3d = static_cast<Node3D*>(child);
        Transform t = Transform::mat4_to_transform(glm::inverse(get_global_model()) * child_3d->get_model());
        child_3d->set_transform(t);
    }
}

void Group3D::add_node(Node3D* node)
{
    add_child(node);
    update_pivot();
}

void Group3D::add_nodes(const std::vector<Node3D*>& nodes)
{
    for (auto& node : nodes) {
        add_node(node);
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
