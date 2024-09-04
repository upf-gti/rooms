#pragma once

#include "base_editor.h"

#include "framework/ui/gizmo_2d.h"

class RoomsRenderer;
class Animation;
class Track;

namespace ui {
    class Inspector;
}

class AnimationEditor : public BaseEditor {

    Gizmo2D gizmo;

    Animation* animation = nullptr;

    Track* current_track = nullptr;

    float current_time = 0.0f;

    /*
        UI
    */

    static uint64_t node_signal_uid;

    bool inspector_dirty = true;
    bool inspector_transform_dirty = false;

    ui::Inspector* inspector = nullptr;
    Viewport3D* inspect_panel_3d = nullptr;

    void init_ui();
    void bind_events();

public:

    AnimationEditor() {};
    AnimationEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;
    void render_gui() override;
};
