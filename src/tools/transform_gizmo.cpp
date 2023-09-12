#include "transform_gizmo.h"

#include <graphics/mesh.h>
#include <graphics/shader.h>
#include <framework/entities/entity_mesh.h>
#include <framework/input.h>
#include <framework/intersections.h>
#include <utils.h>

void TransformGizmo::initialize(const eGizmoType gizmo_use) {
	type = gizmo_use;
	arrow_mesh = new EntityMesh();
	arrow_mesh->set_mesh(Mesh::get("data/meshes/arrow.obj"));
	arrow_mesh->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));

	gizmo_scale = glm::vec3(0.1f, 0.1f, 0.1f);
}

void TransformGizmo::clean() {
	delete arrow_mesh;
}

glm::vec3 TransformGizmo::update(const glm::vec3 &new_position) {
	if (!enabled) {
		return new_position;
	}

	// TODO the other gizmos
	if (type != POSITION_GIZMO) {
		return new_position;
	}

	gizmo_position = new_position;

	glm::vec3 controller_position = Input::get_controller_position(HAND_RIGHT);

	if (!has_graved) {
		// Generate the AABB size of the bounding box of the arrwo mesh
		// Center the pointa ccordingly, and add abit of margin (0.01f) for
		// Grabing all axis in the bottom
		// X axis: 
		glm::vec3 size = gizmo_scale * glm::vec3(mesh_size.y, mesh_size.x, mesh_size.z);
		glm::vec3 box_center = gizmo_position + glm::vec3(-size.x / 2.0f + 0.01f, 0.0f, 0.0f);
		axis_x_selected = intersection::point_AABB(controller_position, box_center, size);

		// Y axis: 
		size = gizmo_scale * mesh_size;
		box_center = gizmo_position + glm::vec3(0.0f, size.y / 2.0f - 0.01f, 0.0f);
		axis_y_selected = intersection::point_AABB(controller_position, box_center, size);

		// Z axis: 
		size = gizmo_scale * glm::vec3(mesh_size.x, mesh_size.z, mesh_size.y);
		box_center = gizmo_position + glm::vec3(0.0f, 0.0f, size.z / 2.0f - 0.01f);
		axis_z_selected = intersection::point_AABB(controller_position, box_center, size);
	}

	// Calculate the movement vector for the gizmo
	if (Input::get_grab_value(HAND_RIGHT) > 0.3f) {
		if (has_graved) {
			glm::vec3 controller_delta = controller_position - prev_controller_position;

			glm::vec3 constraint = { 0.0f, 0.0f, 0.0f };
			if (axis_x_selected) {
				constraint.x = 1.0f;
			}
			if (axis_y_selected) {
				constraint.y = 1.0f;
			}
			if (axis_z_selected) {
				constraint.z = 1.0f;
			}

			gizmo_position += controller_delta * constraint;
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

	if (type == POSITION_GIZMO) {
		arrow_mesh->set_translation(gizmo_position);
		arrow_mesh->scale(gizmo_scale);
		arrow_mesh->set_color(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) + ((axis_y_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();

		arrow_mesh->rotate(0.0174533f * 90.0f, glm::vec3(0.0f, 0.0f, 1.0f));
		arrow_mesh->set_color(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) + ((axis_x_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();

		arrow_mesh->rotate(0.0174533f * 90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
		arrow_mesh->set_color(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) + ((axis_z_selected) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f) : glm::vec4(0.0f)));
		arrow_mesh->render();
	}
}