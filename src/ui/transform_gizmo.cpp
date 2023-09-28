#include "transform_gizmo.h"

#include <graphics/mesh.h>
#include <graphics/shader.h>
#include <framework/entities/entity_mesh.h>
#include <framework/input.h>
#include <framework/intersections.h>
#include <utils.h>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "framework/scene/parse_scene.h"

void TransformGizmo::initialize(const eGizmoType gizmo_use, const glm::vec3 &position) {
	type = gizmo_use;

    if (gizmo_use & POSITION_GIZMO) {
        arrow_mesh = parse_scene("data/meshes/arrow.obj");
    }
    if (gizmo_use & ROTATION_GIZMO) {
        wire_circle_mesh = parse_scene("data/meshes/wired_circle.obj");
    }

	position_gizmo_scale = glm::vec3(0.1f, 0.1f, 0.1f);
    rotation_gizmo_scale = glm::vec3(0.25f, 0.25f, 0.25f);
    gizmo_position = position;
}

void TransformGizmo::clean() {
	delete arrow_mesh;
}

float get_angle(const glm::vec3& v1, const glm::vec3& v2) {
    const float dot = glm::dot((v1), (v2));
    return glm::acos(dot / (v1.length() * v2.length()));
}

glm::vec3 TransformGizmo::update(const glm::vec3 &new_position, const float delta) {
	if (!enabled) {
		return new_position;
	}

	gizmo_position = new_position;

	glm::vec3 controller_position = Input::get_controller_position(HAND_RIGHT);

	if (!has_graved) {
        // POSITION GIZMO TESTS
		// Generate the AABB size of the bounding box of the arrwo mesh
		// Center the pointa ccordingly, and add abit of margin (0.01f) for
		// Grabing all axis in the bottom
		// X axis: 
		glm::vec3 size = position_gizmo_scale * glm::vec3(mesh_size.y, mesh_size.x, mesh_size.z);
		glm::vec3 box_center = gizmo_position + glm::vec3(-size.x / 2.0f + 0.01f, 0.0f, 0.0f);
        position_axis_x_selected = intersection::point_AABB(controller_position, box_center, size);

		// Y axis: 
		size = position_gizmo_scale * mesh_size;
		box_center = gizmo_position + glm::vec3(0.0f, size.y / 2.0f - 0.01f, 0.0f);
        position_axis_y_selected = intersection::point_AABB(controller_position, box_center, size);

		// Z axis: 
		size = position_gizmo_scale * glm::vec3(mesh_size.x, mesh_size.z, mesh_size.y);
		box_center = gizmo_position + glm::vec3(0.0f, 0.0f, size.z / 2.0f - 0.01f);
        position_axis_z_selected = intersection::point_AABB(controller_position, box_center, size);

        const bool any_translation_grabed = position_axis_x_selected || position_axis_y_selected || position_axis_z_selected;

        // ROTATION GIZMO
        rotation_axis_x_selected = !any_translation_grabed && intersection::point_circle(controller_position, gizmo_position, current_rotation * glm::vec3(1.0f, 0.0f, 0.0f), rotation_gizmo_scale.y * 0.25f);
        rotation_axis_y_selected = !rotation_axis_x_selected && intersection::point_circle(controller_position, gizmo_position, current_rotation * glm::vec3(0.0f, 1.0f, 0.0f), rotation_gizmo_scale.x * 0.25f);
        rotation_axis_z_selected = !rotation_axis_y_selected && intersection::point_circle(controller_position, gizmo_position, current_rotation * glm::vec3(0.0f, 0.0f, 1.0f), rotation_gizmo_scale.z * 0.25f);
	}

	// Calculate the movement vector for the gizmo
	if (Input::get_grab_value(HAND_RIGHT) > 0.3f) {
        glm::vec3 controller_delta = controller_position - prev_controller_position;

		if (has_graved) {
            if (type & POSITION_GIZMO) {
                glm::vec3 constraint = { 0.0f, 0.0f, 0.0f };
                if (position_axis_x_selected) {
                    constraint.x = 1.0f;
                }
                if (position_axis_y_selected) {
                    constraint.y = 1.0f;
                }
                if (position_axis_z_selected) {
                    constraint.z = 1.0f;
                }

                gizmo_position += controller_delta * constraint;
            }

            if (type & ROTATION_GIZMO) {
                x_angle = 0.0f, y_angle = 0.0f, z_angle = 0.0f;
                glm::quat z_rot = { 0.0f, 0.0f, 0.0f, 1.0f }, x_rot = { 0.0f, 0.0f, 0.0f, 1.0f }, y_rot = { 0.0f, 0.0f, 0.0f, 1.0f };

                const glm::vec3 new_rotation_pose = glm::normalize(controller_position - gizmo_position);

                if (rotation_axis_y_selected) {
                    const glm::vec2 ref_pos_no_y = { reference_rotation_pose.x, reference_rotation_pose.z };

                    x_angle = -glm::orientedAngle(glm::vec2(new_rotation_pose.x, new_rotation_pose.z), ref_pos_no_y);
                    x_rot = glm::angleAxis(x_angle, reference_rotation * glm::vec3(0.0f, 1.0f, 0.0f));
                }

                if (rotation_axis_x_selected) {
                    const glm::vec2 ref_pos_no_x = { reference_rotation_pose.y, reference_rotation_pose.z };

                    y_angle = glm::orientedAngle(glm::vec2(new_rotation_pose.y, new_rotation_pose.z), ref_pos_no_x);
                    y_rot = glm::angleAxis(y_angle, reference_rotation * glm::vec3(0.0f, 0.0f, 1.0f));
                }

                if (rotation_axis_z_selected) {
                    const glm::vec2 ref_pos_no_z = { reference_rotation_pose.x, reference_rotation_pose.y };

                    z_angle = -glm::orientedAngle(glm::vec2(new_rotation_pose.x, new_rotation_pose.y), ref_pos_no_z);
                    z_rot = glm::angleAxis(z_angle, reference_rotation * glm::vec3(1.0f, 0.0f, 0.0f));
                }

                if (rotation_axis_x_selected || rotation_axis_y_selected || rotation_axis_z_selected) {
                    glm::quat frame_rotation = z_rot * y_rot * x_rot;
                    glm::quat rotation_diff = glm::inverse(reference_rotation) * glm::inverse(frame_rotation);
                    current_rotation = (rotation_diff * reference_rotation);
                }
            }
        } else {
            // Stablish reference to rotation
            if (type & ROTATION_GIZMO) {
                reference_rotation = current_rotation;
                reference_rotation_pose = glm::normalize(controller_position - gizmo_position);
            }
        }

		prev_controller_position = controller_position;
		has_graved = true;
	} else {
		has_graved = false;
	}

	return gizmo_position;
}

void TransformGizmo::render() {
	if (!enabled) {
		return;
	}

	if (type & POSITION_GIZMO) {
		arrow_mesh->set_translation(gizmo_position);
		arrow_mesh->scale(position_gizmo_scale);
		arrow_mesh->set_material_color(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) + ((position_axis_y_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();

		arrow_mesh->rotate(0.0174533f * 90.0f, glm::vec3(0.0f, 0.0f, 1.0f));
		arrow_mesh->set_material_color(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) + ((position_axis_x_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();

		arrow_mesh->rotate(0.0174533f * 90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
		arrow_mesh->set_material_color(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) + ((position_axis_z_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();
	}

    if (type & ROTATION_GIZMO) {
        wire_circle_mesh->set_translation(gizmo_position);
        wire_circle_mesh->rotate(current_rotation);
        wire_circle_mesh->set_material_color(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) + ((rotation_axis_z_selected) ? glm::vec4(1.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
        wire_circle_mesh->scale(rotation_gizmo_scale * 0.5f);
        wire_circle_mesh->render();

        wire_circle_mesh->rotate(0.0174533f * 90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
        wire_circle_mesh->set_material_color(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) + ((rotation_axis_y_selected) ? glm::vec4(1.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
        wire_circle_mesh->scale(glm::vec3(0.98f));
        wire_circle_mesh->render();

        wire_circle_mesh->rotate(0.0174533f * 90.0f, glm::vec3(0.0f, 1.0f, 0.0f));
        wire_circle_mesh->set_material_color(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) + ((rotation_axis_x_selected) ? glm::vec4(1.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
        wire_circle_mesh->scale(glm::vec3(0.98f));
        wire_circle_mesh->render();
    }
}
