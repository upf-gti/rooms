#pragma once

#include "framework/nodes/node_3d.h"


class RoomsEngine;

#define PLAYER_TRANSLATION_SPEED 1.6f
#define PLAYER_ROTATION_SPEED 1.8f

class PlayerNode : public Node3D {
    RoomsEngine *engine = nullptr;

    bool was_trigger_pressed = false;
    glm::vec3 prev_lcontroller_position = {};

public:
    PlayerNode();

    void update(float delta_time);
};
