#include "cursor.h"

#include "framework/nodes/ui.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"
#include "framework/ui/context_2d.h"

#include "graphics/renderer.h"

#include "glm/gtx/quaternion.hpp"

#include "imgui.h"

namespace ui {

    void Cursor::load()
    {
        size = { 32.f, 32.f };

        c_default = new Image2D("default_cursor", "data/textures/cursors/pointer_d.png", size, CURSOR);
        c_pointer = new Image2D("default_cursor", "data/textures/cursors/dot_large.png", size, CURSOR);
        c_resize_ew = new Image2D("default_cursor", "data/textures/cursors/resize_c_horizontal.png", size, CURSOR);
        c_resize_ns = new Image2D("default_cursor", "data/textures/cursors/resize_c_vertical.png", size, CURSOR);
        c_picker = new Image2D("default_cursor", "data/textures/cursors/drawing_picker.png", size * 0.5f, CURSOR);
        c_disabled = new Image2D("default_cursor", "data/textures/cursors/disabled.png", size, CURSOR);

        current = c_default;

        if (Renderer::instance->get_openxr_available()) {
            HContainer2D* root_cursor = new HContainer2D("root_cursor", { 0.0f, 0.0f });
            root_cursor->add_child(current);
            cursor_3d = new Viewport3D(root_cursor);
        }
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

    void Cursor::update(float delta_time)
    {
        if (Renderer::instance->get_openxr_available()) {

            must_render_xr_cursor = false;

            if (!Context2D::any_hover()) {
                return;
            }

            Node2D* hovered = Context2D::get_hover();

            glm::mat4x4 m(1.0f);
            m = glm::translate(m, Context2D::get_xr_world_position());
            m = m * glm::toMat4(glm::quat_cast(hovered->get_global_viewport_model()));

            cursor_3d->set_model(m);

            cursor_3d->update(delta_time);

            // must_render_xr_cursor = true;
        }
    }

    void Cursor::render()
    {
        if (!current) {
            return;
        }

        if (Renderer::instance->get_openxr_available()) {

            if (must_render_xr_cursor) {
                cursor_3d->render();
            }

        } else {

            auto& io = ImGui::GetIO();
            if (io.WantCaptureMouse) {
                return;
            }

            ImGui::SetMouseCursor(ImGuiMouseCursor_None);

            auto webgpu_context = Renderer::instance->get_webgpu_context();

            glm::vec2 mouse_pos = Input::get_mouse_position();
            mouse_pos.y = webgpu_context->render_height - mouse_pos.y;

            glm::vec2 cursor_position = { mouse_pos.x - size.x * 0.25f, mouse_pos.y - size.y * 0.75f };

            current->set_translation(cursor_position);

            current->render();
        }
    }
}
