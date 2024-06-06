#include "cursor.h"

#include "framework/nodes/ui.h"
#include "framework/input.h"
#include "framework/ui/context_2d.h"

#include "graphics/renderer.h"

namespace ui {

    void Cursor::load()
    {
        size = { 32.f, 32.f };

        c_default = new Image2D("default_cursor", "data/textures/cursors/pointer_d.png", size);
        c_pointer = new Image2D("default_cursor", "data/textures/cursors/dot_large.png", size);
        c_resize_ew = new Image2D("default_cursor", "data/textures/cursors/resize_c_horizontal.png", size);
        c_resize_ns = new Image2D("default_cursor", "data/textures/cursors/resize_c_vertical.png", size);
        c_picker = new Image2D("default_cursor", "data/textures/cursors/drawing_picker.png", size * 0.5f);
        c_disabled = new Image2D("default_cursor", "data/textures/cursors/disabled.png", size);

        // Set as cursors for rendering priority
        c_default->set_priority(CURSOR);
        c_pointer->set_priority(CURSOR);
        c_resize_ew->set_priority(CURSOR);
        c_resize_ns->set_priority(CURSOR);
        c_picker->set_priority(CURSOR);
        c_disabled->set_priority(CURSOR);

        current = c_default;
    }

    void Cursor::set(int type)
    {
        switch (type)
        {
        case MOUSE_CURSOR_NONE:
            current = nullptr;
            break;
        case MOUSE_CURSOR_DEFAULT:
            current = c_default;
            break;
        case MOUSE_CURSOR_POINTER:
            current = c_pointer;
            break;
        case MOUSE_CURSOR_RESIZE_EW:
            current = c_resize_ew;
            break;
        case MOUSE_CURSOR_RESIZE_NS:
            current = c_resize_ns;
            break;
        case MOUSE_CURSOR_PICKER:
            current = c_picker;
            break;
        case MOUSE_CURSOR_DISABLED:
            current = c_disabled;
            break;
        default:
            assert(0 && "Undefined type of cursor!");
        }
    }

    void Cursor::render()
    {
        if (!current) {
            return;
        }

        glm::vec2 cursor_position;

        if (Renderer::instance->get_openxr_available()) {

            if (!Context2D::any_hover()) {
                return;
            }

            cursor_position = Context2D::get_xr_position();

        } else {
            auto webgpu_context = Renderer::instance->get_webgpu_context();

            glm::vec2 mouse_pos = Input::get_mouse_position();
            mouse_pos.y = webgpu_context->render_height - mouse_pos.y;

            cursor_position = { mouse_pos.x - size.x * 0.25f, mouse_pos.y - size.y * 0.75f };
        }

        current->set_translation(cursor_position);

        current->render();
    }
}
