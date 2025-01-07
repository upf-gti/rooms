#include "player_node.h"

#include "framework/nodes/node_factory.h"
#include "framework/math/intersections.h"
#include "graphics/renderers/rooms_renderer.h"
#include "framework/input.h"
#include "engine/rooms_engine.h"

#ifdef XR_SUPPORT
#include "xr/openxr_context.h"
#endif

PlayerNode::PlayerNode(RoomsEngine* engine_ref) : Node3D()
{
    node_type = "PlayerNode";

#ifdef XR_SUPPORT
    if (RoomsRenderer::instance->get_openxr_available()) {
        // Sets this node 's transform as teh origin for the OpenXR
        Renderer::instance->get_openxr_context()->root_transform = &transform;
    }
#endif

    engine = engine_ref;
}

void PlayerNode::update(float delta_time)
{
    if (engine->get_current_editor_type() != SCENE_EDITOR) {
        return;
    }

    // TODO: add inercia or something to make it feel better
    if (Input::is_trigger_pressed(HAND_LEFT)) {
        if (!was_trigger_pressed) {
            prev_lcontroller_position = Input::get_controller_position(HAND_LEFT, POSE_GRIP, false);
            was_trigger_pressed = true;
        } else {
            const glm::vec3 curr_lcontroller_pos = Input::get_controller_position(HAND_LEFT, POSE_GRIP, false);

            transform.translate((prev_lcontroller_position - curr_lcontroller_pos));

            prev_lcontroller_position = curr_lcontroller_pos;
        }
    } else {
        was_trigger_pressed = false;
    }
}
