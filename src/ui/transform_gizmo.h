#pragma once

#include <glm/glm.hpp>

enum eGizmoType : uint8_t {
	POSITION_GIZMO = 0,
	ROTATION_GIZMO,
	POSITION_ROTATION_GIZMO
};

class EntityMesh;

/*
	TRANSFORM GIZMO COMPONENT
	- Declare the kind of gizmo that you want (for now, only Position)
	- On update, give it the base position, and get the new position of
	  the gizmo, in order to use it for the parent.
*/

class TransformGizmo {

	eGizmoType				type;
	bool					enabled = true;
	
	EntityMesh*				arrow_mesh = nullptr;

	glm::vec3				prev_controller_position;
	bool					has_graved = false;

	glm::vec3               gizmo_position = {};
	glm::vec3               gizmo_scale = {};

	glm::vec3               mesh_size = { 0.300f, 1.7f, 0.300f };

	bool                    axis_x_selected = false;
	bool                    axis_y_selected = false;
	bool                    axis_z_selected = false;

public:

	void			initialize(const eGizmoType gizmo_use);
	void			clean();

	glm::vec3		update(const glm::vec3& new_position);
	void			render();
};