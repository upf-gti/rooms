#pragma once

#include "base_editor.h"

#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Track;

class AnimationEditor : public BaseEditor {

    Gizmo2D gizmo;

    Animation* animation = nullptr;

    Track* current_track = nullptr;

    float current_time = 0.0f;

public:

    AnimationEditor() {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
