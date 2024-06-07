#pragma once

#include "includes.h"

#include "glm/vec2.hpp"

class Viewport3D;

namespace ui {

    class Image2D;

    enum MouseCursor
    {
        MOUSE_CURSOR_NONE = -1,
        MOUSE_CURSOR_DEFAULT,
        MOUSE_CURSOR_CIRCLE,
        MOUSE_CURSOR_POINTER,
        MOUSE_CURSOR_RESIZE_NS,
        MOUSE_CURSOR_RESIZE_EW,
        MOUSE_CURSOR_PICKER,
        MOUSE_CURSOR_GRAB,
        MOUSE_CURSOR_DISABLED,
        MOUSE_CURSOR_COUNT
    };

    class Cursor {

        glm::vec2 size = {};

        Image2D* current = nullptr;

        // Support for vr cursor

        bool is_xr = false;
        Viewport3D* cursor_3d = nullptr;
        bool must_render_xr_cursor = false;

        // Pre-load different cursors

        Image2D* cursors[MOUSE_CURSOR_COUNT];

    public:

        Cursor() {};

        void load();
        void set(int type);

        void update(float delta_time);
        void render();
    };
}
