#pragma once

#include "tools/module.h"

#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Track;

class AnimationModule : public Module {

    Gizmo2D gizmo;

    Animation* animation = nullptr;

    Track* current_track = nullptr;

    float current_time = 0.0f;

public:

    AnimationModule() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
