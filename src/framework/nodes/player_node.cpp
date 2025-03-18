#include "player_node.h"

#include "framework/nodes/node_factory.h"
#include "framework/math/intersections.h"
#include "graphics/renderers/rooms_renderer.h"
#include "framework/input.h"
#include "engine/rooms_engine.h"

#ifdef XR_SUPPORT
#include "xr/openxr_context.h"
#endif
#include <glm/gtx/vector_angle.hpp>

REGISTER_NODE_CLASS(PlayerNode)

PlayerNode::PlayerNode() : Node3D()
{
    node_type = "PlayerNode";

#ifdef XR_SUPPORT
    if (RoomsRenderer::instance->get_openxr_available()) {
        // Sets this node 's transform as teh origin for the OpenXR
        Renderer::instance->get_openxr_context()->root_transform = &transform;
    }
#endif

    engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
}

void PlayerNode::update(float delta_time)
{
    if (engine->get_current_editor_type() != SCENE_EDITOR) {
        return;
    }

    // TODO: add inercia or something to make it feel better
    if (Input::is_trigger_pressed(HAND_LEFT)) {
        if (!was_trigger_pressed) {
            // Get the controllers in world position
            prev_lcontroller_position = Input::get_controller_position(HAND_LEFT, POSE_GRIP, false);
            was_trigger_pressed = true;
        } else {
            const glm::vec3 curr_lcontroller_pos = Input::get_controller_position(HAND_LEFT, POSE_GRIP, false);

            if (Input::is_grab_pressed(HAND_LEFT)) {
                const glm::vec3 head_pos = glm::inverse(transform.get_model()) * glm::vec4(transform.get_position(), 1.0);

                const glm::vec2 head_pos_2d = glm::vec2(head_pos.x, head_pos.y);
                const glm::vec2 prev_lcontroller_pos_2d = glm::vec2(prev_lcontroller_position.x, prev_lcontroller_position.y);
                const glm::vec2 curr_lcontroller_pos_2d = glm::vec2(curr_lcontroller_pos.x, curr_lcontroller_pos.y);

                glm::vec2 curr_vec = glm::normalize(curr_lcontroller_pos_2d - head_pos_2d);
                glm::vec2 prev_vec = glm::normalize(prev_lcontroller_pos_2d - head_pos_2d);

                float d = glm::length(curr_vec - prev_vec);

                float rot_angle = glm::orientedAngle(curr_vec, prev_vec);

                // For a bit more rotation speed
                rot_angle *= PLAYER_ROTATION_SPEED;

                glm::quat rot_diff = glm::normalize(glm::angleAxis(rot_angle, glm::vec3(0.0f, 1.0f, 0.0f)));

                transform.rotate(rot_diff);
            }
            else {
                // TODO: This could be speed dependant
                const glm::vec3 movement = prev_lcontroller_position - curr_lcontroller_pos;
                transform.translate(glm::normalize(movement) * glm::length(movement) * PLAYER_TRANSLATION_SPEED);
            }

            prev_lcontroller_position = curr_lcontroller_pos;
        }
    } else {
        was_trigger_pressed = false;
    }
}
