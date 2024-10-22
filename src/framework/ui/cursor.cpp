#include "cursor.h"

#include "framework/nodes/panel_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"
#include "framework/ui/io.h"

#include "graphics/renderer.h"

#include "glm/gtx/quaternion.hpp"

#include "imgui.h"

namespace ui {

    void Cursor::initialize()
    {
        is_xr = Renderer::instance->get_openxr_available();

        size = is_xr ? glm::vec2(42.f) : glm::vec2(32.f);

        cursors[MOUSE_CURSOR_DEFAULT] = new Image2D("default_cursor", "data/textures/cursors/pointer_d.png", size, CURSOR);
        cursors[MOUSE_CURSOR_CIRCLE] = new Image2D("circle_cursor", "data/textures/cursors/dot_large.png", size, CURSOR);
        cursors[MOUSE_CURSOR_POINTER] = new Image2D("pointer_cursor", "data/textures/cursors/hand_small_point.png", size, CURSOR);
        cursors[MOUSE_CURSOR_RESIZE_EW] = new Image2D("resize_ew_cursor", "data/textures/cursors/resize_c_horizontal.png", size * 0.75f, CURSOR);
        cursors[MOUSE_CURSOR_RESIZE_NS] = new Image2D("resize_ns_cursor", "data/textures/cursors/resize_c_vertical.png", size * 0.75f, CURSOR);
        cursors[MOUSE_CURSOR_PICKER] = new Image2D("picker_cursor", "data/textures/cursors/drawing_picker.png", size * 0.5f, CURSOR);
        cursors[MOUSE_CURSOR_GRAB] = new Image2D("grab_cursor", "data/textures/cursors/hand_small_closed.png", size * 0.5f, CURSOR);
        cursors[MOUSE_CURSOR_DISABLED] = new Image2D("disabled_cursor", "data/textures/cursors/disabled.png", size * 0.75f, CURSOR);

        // Set some offsets
        {
            cursors_data[MOUSE_CURSOR_DEFAULT].offset = size * 0.2f;
            cursors_data[MOUSE_CURSOR_CIRCLE].offset = size * 0.5f;
            cursors_data[MOUSE_CURSOR_POINTER].offset = { size.x * 0.25f, size.y * 0.15f };
            cursors_data[MOUSE_CURSOR_RESIZE_EW].offset = { size.x * 0.35f, size.y * 0.3f };
            cursors_data[MOUSE_CURSOR_RESIZE_NS].offset = { size.x * 0.35f, size.y * 0.3f };
            cursors_data[MOUSE_CURSOR_DISABLED].offset = size * 0.25f;
        }

        current = cursors[MOUSE_CURSOR_DEFAULT];

        // Set all visibility as false by default
        for (uint8_t i = 0u; i < MOUSE_CURSOR_COUNT; ++i) {
            cursors[i]->set_visibility(false);
        }

        if (is_xr) {

            HContainer2D* root_cursor = new HContainer2D("root_cursor", { 0.0f, 0.0f });

            // Add all of them to the viewport 3d

            for (uint8_t i = 0u; i < MOUSE_CURSOR_COUNT; ++i) {
                root_cursor->add_child(cursors[i]);
            }

            cursor_3d = root_cursor->get_xr_viewport();
        }
    }

    void Cursor::set(int type)
    {
        for (uint8_t i = 0u; i < MOUSE_CURSOR_COUNT; ++i) {
            cursors[i]->set_visibility(false);
        }

        if (type == MOUSE_CURSOR_NONE || type >= MOUSE_CURSOR_COUNT) {
            current = nullptr;
            return;
        }

        current_type = type;

        current = cursors[type];

        current->set_visibility(true);
    }

    void Cursor::update(float delta_time)
    {
        // IO data could have changed by now, so check for updates

        if (IO::any_hover()) {

            if (IO::is_any_hover_type({ PANEL_BUTTON, BUTTON, TEXTURE_BUTTON, SELECTOR_BUTTON, COMBO_BUTTON, SUBMENU })) {
                set(ui::MOUSE_CURSOR_POINTER);
            }
            else if (IO::is_hover_type(HSLIDER)) {
                set(ui::MOUSE_CURSOR_RESIZE_EW);
            }
            else if (IO::is_hover_type(VSLIDER)) {
                set(ui::MOUSE_CURSOR_RESIZE_NS);
            }

            if (IO::is_hover_disabled()) {
                set(ui::MOUSE_CURSOR_DISABLED);
            }
        }

        if (is_xr) {

            must_render_xr_cursor = false;

            if (!IO::any_hover()) {
                return;
            }

            Node2D* hovered = IO::get_hover();

            glm::mat4x4 m(1.0f);
            m = glm::translate(m, IO::get_xr_world_position());
            m = m * glm::toMat4(glm::quat_cast(hovered->get_global_viewport_model()));
            m = glm::translate(m, glm::vec3(-cursors_data[current_type].offset * hovered->get_scale(), 0.0f));

            cursor_3d->set_transform(Transform::mat4_to_transform(m));

            cursor_3d->update(delta_time);

            must_render_xr_cursor = true;
        }
    }

    void Cursor::render()
    {
#ifdef __EMSCRIPTEN__
        return;
#endif

        if (!current) {
            return;
        }

        if (is_xr) {

            if (must_render_xr_cursor) {
                cursor_3d->render();
            }

        } else {

            auto& io = ImGui::GetIO();
            if (io.WantCaptureMouse) {
                return;
            }

            ImGui::SetMouseCursor(ImGuiMouseCursor_None);

            glm::vec2 mouse_pos = Input::get_mouse_position();
            glm::vec2 cursor_position = mouse_pos - cursors_data[current_type].offset;

            current->set_position(cursor_position);

            current->render();
        }
    }
}
