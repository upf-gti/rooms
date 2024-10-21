#pragma once

#include "includes.h"

#include "graphics/renderer.h"
#include "graphics/edit.h"

#include "raymarching_renderer.h"

// #define DISABLE_RAYMARCHER

class RoomsRenderer : public Renderer {

    RaymarchingRenderer raymarching_renderer;

    float last_evaluation_time = 0.0f;

public:

    RoomsRenderer();
    ~RoomsRenderer();

    virtual int pre_initialize(GLFWwindow* window, bool use_mirror_screen = false) override;
    virtual int initialize() override;
    virtual int post_initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    RaymarchingRenderer* get_raymarching_renderer() { return &raymarching_renderer; }
    float get_last_evaluation_time() { return last_evaluation_time; }

    /*
    *   Edits
    */

    void change_stroke(const StrokeParameters& params, const uint32_t index = 1u) {
        raymarching_renderer.change_stroke(params, index);

    }

    void push_edit(Edit edit) {
        raymarching_renderer.push_edit(edit);
    };

    void push_edit_list(std::vector<Edit> &edits) {
#ifndef DISABLE_RAYMARCHER
        raymarching_renderer.push_edit_list(edits);
#endif
    };

    void push_preview_edit_list(std::vector<Edit>& edits) {
#ifndef DISABLE_RAYMARCHER
        for (uint32_t i = 0u; i < edits.size(); i++) {
            raymarching_renderer.add_preview_edit(edits[i]);
        }
#endif
    }

    void undo() {
        raymarching_renderer.undo();
    }

    void redo() {
        raymarching_renderer.redo();
    }

    inline void set_preview_edit(const Edit& stroke) {
        raymarching_renderer.add_preview_edit(stroke);
    }
};
