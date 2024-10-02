#include "group_3d.h"

#include "framework/math/intersections.h"

uint32_t Group3D::last_uid = 0;

Group3D::Group3D()
{
    node_type = "Group3D";
    name = "Group" + std::to_string(last_uid++);
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
