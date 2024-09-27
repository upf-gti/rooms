#include "group_3d.h"

#include "framework/math/intersections.h"

Group3D::Group3D()
{
    node_type = "Group3D";
}

bool Group3D::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance)
{
    for (auto node : get_children()) {

        Node3D* node_3d = dynamic_cast<Node3D*>(node);

        if (!node_3d) {
            continue;
        }

        float node_distance = 1e10f;

        if (node_3d->test_ray_collision(ray_origin, ray_direction, node_distance)) {
            return true;
        }
    }

    return false;
}
