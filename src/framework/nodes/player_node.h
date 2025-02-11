#pragma once

#include "framework/nodes/node_3d.h"


class RoomsEngine;

class PlayerNode : public Node3D {
    RoomsEngine *engine = nullptr;

    bool was_trigger_pressed = false;
    glm::vec3 prev_lcontroller_position = {};

    glm::quat prev_lcontroller_rotation = {};

public:
    PlayerNode();

    void update(float delta_time);
};
