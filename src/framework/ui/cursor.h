#pragma once

#include "includes.h"

#include "glm/vec2.hpp"

namespace ui {

    class Image2D;

    enum MouseCursor
    {
        MOUSE_CURSOR_NONE = -1,
        MOUSE_CURSOR_DEFAULT,
        MOUSE_CURSOR_POINTER,
        MOUSE_CURSOR_RESIZE_NS,
        MOUSE_CURSOR_RESIZE_EW,
        MOUSE_CURSOR_PICKER,
        //MOUSE_CURSOR_GRAB,
        MOUSE_CURSOR_DISABLED,
        MOUSE_CURSOR_COUNT
    };

    class Cursor {

        glm::vec2 size = {};

        Image2D* current = nullptr;

        // Pre-load different cursors

        Image2D* c_default      = nullptr;
        Image2D* c_pointer      = nullptr;
        Image2D* c_resize_ew    = nullptr;
        Image2D* c_resize_ns    = nullptr;
        Image2D* c_picker       = nullptr;
        Image2D* c_disabled     = nullptr;
        // Image2D* c_grab         = nullptr;

    public:

        Cursor() {};

        void load();

        void set(int type);

        void render();
    };
}
