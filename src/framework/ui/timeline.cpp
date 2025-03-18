#include "timeline.h"

#include "framework/input.h"
#include "framework/ui/io.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/animation/animation.h"
#include "framework/math/intersections.h"

#include "engine/engine.h"

#include "graphics/renderer.h"

#include "glm/gtx/quaternion.hpp"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "spdlog/spdlog.h"

namespace ui {

    Timeline::Timeline(const TimelineDesc& desc)
        : Panel2D(desc.name, desc.position, { 0.0f, 0.0f }, ui::CREATE_3D), panel_size(desc.size), padding(desc.padding)
    {
        float inner_width = panel_size.x - padding * 2.0f;
        float inner_height = panel_size.y - padding * 2.0f;

        // Used when hovering a keyframe..
        set_priority(BUTTON);

        custom_on_close = desc.close_fn;
        custom_on_insert_keyframe = desc.insert_keyframe_fn;
        custom_on_edit_keyframe = desc.edit_keyframe_fn;
        custom_on_duplicate_keyframe = desc.duplicate_keyframe_fn;
        custom_on_move_keyframe = desc.move_keyframe_fn;
        custom_on_delete_keyframe = desc.delete_keyframe_fn;

        root = new ui::XRPanel(name + "_background", { 0.0f, 0.f }, panel_size, 0u, panel_color);
        add_child(root);

        ui::VContainer2D* column = new ui::VContainer2D(name + "_column", glm::vec2(padding));
        column->set_fixed_size({ inner_width, inner_height });
        root->add_child(column);

        // Title
        float button_start = padding * 2.0f;
        float button_space = 36.0f; // size + Xpadding
        float title_text_scale = 22.0f;
        float title_y_corrected = desc.title_height * 0.5f - title_text_scale * 0.5f;
        ui::Container2D* title_container = new ui::Container2D(name + "_title", { 0.0f, 0.0f }, { inner_width - padding * 0.4f, desc.title_height });
        title = new ui::Text2D(desc.title.empty() ? "Inspector" : desc.title, { 0.0f, title_y_corrected }, title_text_scale, ui::TEXT_CENTERED | ui::SKIP_TEXT_RECT);
        auto insert_button = new ui::TextureButton2D(name + "@insert_keyframe", { "data/textures/add.png", 0u, { button_start, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Insert" });
        auto edit_button = new ui::TextureButton2D(name + "@edit_keyframe", { "data/textures/edit.png", 0u, { button_start + button_space, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Edit" });
        auto duplicate_button = new ui::TextureButton2D(name + "@duplicate_keyframe", { "data/textures/duplicate_key.png", 0u, { button_start + button_space * 2.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Duplicate" });
        auto delete_button = new ui::TextureButton2D(name + "@delete_keyframe", { "data/textures/delete.png", 0u, { button_start + button_space * 3.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Delete" });
        auto close_button = new ui::TextureButton2D(name + "@close_timeline", { "data/textures/cross.png", 0u, { inner_width - padding * 4.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Close" });
        time_text = new ui::Text2D("0.0", { button_start + button_space * 4.0f + 2.0f, title_y_corrected + 8.0f }, 20.f, ui::SKIP_TEXT_RECT);
        title_container->add_childs({ insert_button, edit_button, duplicate_button, delete_button, time_text, title, close_button });
        column->add_child(title_container);

        Node::bind(name + "@insert_keyframe", [&](const std::string& sg, void* data) { on_insert_keyframe(); });
        Node::bind(name + "@edit_keyframe", [&](const std::string& sg, void* data) { on_edit_keyframe(); });
        Node::bind(name + "@duplicate_keyframe", [&](const std::string& sg, void* data) { on_duplicate_keyframe(); });
        Node::bind(name + "@delete_keyframe", [&](const std::string& sg, void* data) { on_delete_keyframe(); });
        Node::bind(name + "@close_timeline", [&](const std::string& sg, void* data) { on_close_timeline(); });

        render_background = false;

        // Body row
        body = new ui::Container2D(name + "_body", { 0.0f, 0.0f });
        body->set_fixed_size({ inner_width, panel_size.y - desc.title_height - padding * 3.0f });
        column->add_child(body);

        playhead = new ui::Panel2D(name + "_cursor", { inner_width * 0.5f,0.0f }, { 1.0f, panel_size.y - desc.title_height - padding * 3.0f }, 0u, Color(1.0f, 1.0f, 1.0f, 0.25f));
        playhead->set_priority(BUTTON_MARK);
        body->add_child(playhead);

        keyframe_size = { 16.0f, 48.0f };

        quad_surface = new Surface();
        quad_surface->create_quad(keyframe_size.x, keyframe_size.y, true);

        frame_mesh = generate_keyframe_mesh(colors::GRAY);
        frame_mesh_hovered = generate_keyframe_mesh(colors::WHITE);
        frame_mesh_selected = generate_keyframe_mesh(Color(1.0f, 0.136f, 0.0f, 1.0f));

        if (Renderer::instance->get_xr_available()) {
            disable_2d();
        }
    }

    Timeline::~Timeline()
    {
        delete quad_surface;
        delete frame_mesh;
        delete frame_mesh_hovered;
        delete frame_mesh_selected;
    }

    MeshInstance* Timeline::generate_keyframe_mesh(const Color& color)
    {
        Material* material = new Material();
        material->set_color(color);
        material->set_type(MATERIAL_UNLIT);
        material->set_is_2D(true);
        material->set_transparency_type(ALPHA_BLEND);
        material->set_priority(BUTTON);
        material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries, material));

        if (Renderer::instance->get_xr_available()) {
            material->set_is_2D(false);
        }

        MeshInstance* mesh_instance = new MeshInstance();
        mesh_instance->add_surface(quad_surface);
        mesh_instance->set_surface_material_override(mesh_instance->get_surface(0), material);

        return mesh_instance;
    }

    void Timeline::select_keyframe(TimelineKeyframe* key)
    {
        select_keyframe_by_index(key ? key->index : -1);
    }

    void Timeline::select_keyframe_by_index(int index)
    {
        auto selected_key = get_selected_key();

        if (selected_key) {
            selected_key->selected = false;
        }

        if (index != -1) {
            auto& new_key = keyframes[index];
            new_key.selected = true;
            current_time = new_key.time;
        }

        selected_key_index = index;

        time_dirty = true;
    }

    void Timeline::render()
    {
        if (!visibility)
            return;

        glm::vec2 scale = get_scale();
        glm::vec2 offset = glm::vec2(playhead->get_translation().x, body->get_translation().y + padding * scale.y);
        offset.x -= time_to_x(current_time) * scale.x;

        for (auto& key : keyframes) {
            glm::vec2 position = offset + glm::vec2(time_to_x(key.time), keyframe_size.y * 0.5f) * scale;

            if (position.x < (body->get_translation().x + padding * scale.x) || (position.x + keyframe_size.x * scale.x) >(body->get_translation().x + body->fixed_size.x * scale.x)) {
                continue;
            }

            glm::mat4x4 model = glm::translate(glm::mat4x4(1.0f), glm::vec3(position, -CURSOR * 3e-5));
            model = glm::scale(model, glm::vec3(scale, 1.0f));
            model = get_global_viewport_model() * model;
            Renderer::instance->add_renderable(key.selected ? frame_mesh_selected :
                key.hovered ? frame_mesh_hovered : frame_mesh, model);

            // reset input stuff
            key.hovered = false;
        }

        Node2D::render();
    }

    void Timeline::update(float delta_time)
    {
        if (!visibility) {
            return;
        }

        if ((IO::get_hover() == root) && Input::was_grab_pressed(HAND_RIGHT)) {
            grabbing = true;
        }

        if (Input::was_grab_released(HAND_RIGHT)) {
            grabbing = false;
        }

        root->set_priority(PANEL);

        auto renderer = Renderer::instance;

        if (renderer->get_xr_available()) {

            if (!placed) {
                glm::mat4x4 m(1.0f);
                glm::vec3 eye = renderer->get_camera_eye();
                glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.5f;

                m = glm::translate(m, new_pos);
                m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                set_xr_transform(Transform::mat4_to_transform(m));
                placed = true;
            }
            else if (grabbing) {

                Transform raycast_transform = Transform::mat4_to_transform(Input::get_controller_pose(HAND_RIGHT, POSE_AIM));
                const glm::vec3& forward = raycast_transform.get_front();

                glm::mat4x4 m(1.0f);
                glm::vec3 eye = raycast_transform.get_position();

                auto webgpu_context = Renderer::instance->get_webgpu_context();
                float width = static_cast<float>(webgpu_context->render_width);
                float height = static_cast<float>(webgpu_context->render_height);
                glm::vec2 grab_offset = glm::vec2(last_grab_position.x, panel_size.y - last_grab_position.y) / glm::vec2(width, height);
                glm::vec3 new_pos = eye + forward * last_grab_distance;

                m = glm::translate(m, new_pos);
                m = m * glm::toMat4(get_rotation_to_face(new_pos, renderer->get_camera_eye(), { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                m = glm::translate(m, -glm::vec3(grab_offset, 0.0f));
                set_xr_transform(Transform::mat4_to_transform(m));

                root->set_priority(DRAGGABLE);
            }
        }

        // zooming
        {
            float w_delta = Input::get_mouse_wheel_delta();
            zoom -= w_delta * 0.10f;
            zoom = glm::clamp(zoom, 0.3f, 3.0f);
        }

        time_dirty = false;
        on_hover = false;

        // keyframes input
        sInputData keyframe_input_data = get_input_data();

        // timeline panel input
        const sInputData& root_input_data = root->get_input_data(true);

        // is any keyframe hovered
        if (keyframe_input_data.was_released) {
            moving_key = false;
        }

        if (keyframe_input_data.is_hovered) {
            IO::push_input(this, keyframe_input_data);

            if (keyframe_input_data.is_pressed) {
                if (!moving_key) {
                    last_move_positionX = root_input_data.local_position.x;
                }
                moving_key = true;
            }
        }

        if ((selected_key_index != -1) && moving_key) {

            if (root_input_data.was_pressed) {
                IO::set_focus(root);
            }

            if (root_input_data.was_released) {
                IO::set_focus(nullptr);
            }

            float move_dt = (last_move_positionX - root_input_data.local_position.x);
            float dt_time = x_to_time(move_dt);
            float size_offset_time = x_to_time(keyframe_size.x);

            if (glm::abs(dt_time) > 0.0f) {
                uint32_t idx = selected_key_index;
                auto key = get_selected_key();
                float prev_time = (idx > 0 ? keyframes[idx - 1u].time : 0.0f) + size_offset_time;
                float max_allowed_time = 1e10f;
                float next_time = ((idx == (keyframes.size() - 1u)) ? max_allowed_time : keyframes[idx + 1u].time) - size_offset_time;

                key->time -= dt_time;
                key->time = glm::clamp(key->time, prev_time, next_time);

                current_time = key->time;

                if (custom_on_move_keyframe) {
                    custom_on_move_keyframe(this, key->index);
                }
            }

            last_move_positionX = root_input_data.local_position.x;
        }

        // Horizontal scrolling
        if (root_input_data.is_hovered && !moving_key)
        {
            float scroll_dt = 0.0f;

            if (root_input_data.was_pressed) {
                IO::set_focus(root);
            }

            if (root_input_data.is_pressed && !IO::is_focus_type(HSLIDER)) {
                scroll_dt += (last_scroll_positionX - root_input_data.local_position.x);
            }

            if (root_input_data.was_released) {
                IO::set_focus(nullptr);
            }

            last_scroll_positionX = root_input_data.local_position.x;

            float dt_time = x_to_time(scroll_dt);

            float new_time = current_time + dt_time;
            new_time = glm::clamp(new_time, 0.0f, keyframes.back().time);
            time_dirty |= (new_time != current_time);
            current_time = new_time;

            if (renderer->get_xr_available() && grabbing && Input::was_grab_pressed(HAND_RIGHT)) {
                last_grab_position = root_input_data.local_position;
                last_grab_distance = root_input_data.ray_distance;
            }
        }

        std::string s = std::to_string(current_time);
        size_t idx = s.find('.') + 1;
        time_text->set_text(s.substr(0, idx + 2));

        Node2D::update(delta_time);
    }

    sInputData Timeline::get_input_data(bool ignore_focus)
    {
        glm::vec2 scale = get_scale();
        glm::vec2 offset = glm::vec2(playhead->get_translation().x, body->get_translation().y + padding * scale.y);
        offset.x -= time_to_x(current_time) * scale.x;

        sInputData data = {};

        for (auto& key : keyframes) {
            glm::vec2 position = offset + glm::vec2(time_to_x(key.time), keyframe_size.y * 0.5f) * scale;

            if (position.x < (body->get_translation().x + padding * scale.x) || (position.x + keyframe_size.x * scale.x) >(body->get_translation().x + body->fixed_size.x * scale.x)) {
                continue;
            }

            data = {};

            Material* material = frame_mesh->get_surface_material_override(frame_mesh->get_surface(0));

            if (material->get_is_2D())
            {
                const glm::vec2& mouse_pos = Input::get_mouse_position();
                const glm::vec2& min = position - keyframe_size * 0.5f;
                const glm::vec2& max = min + keyframe_size;

                data.is_hovered = mouse_pos.x >= min.x && mouse_pos.y >= min.y && mouse_pos.x <= max.x && mouse_pos.y <= max.y;

                const glm::vec2& local_mouse_pos = mouse_pos - position;
                data.local_position = glm::vec2(local_mouse_pos.x, keyframe_size.y - local_mouse_pos.y);
            }
            else {

                glm::vec3 ray_origin;
                glm::vec3 ray_direction;

                Engine::instance->get_scene_ray(ray_origin, ray_direction);

                // Quad
                uint8_t priority = class_type;
                glm::mat4x4 model = glm::translate(glm::mat4x4(1.0f), glm::vec3(position - keyframe_size * 0.5f * get_scale(), -priority * 1e-4));
                model = get_global_viewport_model() * model;

                glm::vec3 quad_position = model[3];
                glm::quat quad_rotation = glm::quat_cast(model);
                glm::vec2 quad_size = keyframe_size * get_scale();

                float collision_dist;
                glm::vec3 intersection_point;
                glm::vec3 local_intersection_point;

                data.is_hovered = intersection::ray_quad(
                    ray_origin,
                    ray_direction,
                    quad_position,
                    quad_size,
                    quad_rotation,
                    intersection_point,
                    local_intersection_point,
                    collision_dist,
                    false
                );

                if (data.is_hovered) {
                    if (Renderer::instance->get_xr_available()) {
                        data.ray_intersection = intersection_point;
                        data.ray_distance = collision_dist;
                    }

                    glm::vec2 local_pos = glm::vec2(local_intersection_point) / get_scale();
                    data.local_position = glm::vec2(local_pos.x, size.y - local_pos.y);
                }
            }

            data.was_pressed = data.is_hovered && was_input_pressed();

            if (data.was_pressed) {
                pressed_inside = true;
            }

            data.was_released = was_input_released();

            if (data.was_released) {
                if (data.is_hovered) {
                    select_keyframe(&key);
                }
                pressed_inside = false;
            }

            data.is_pressed = pressed_inside && is_input_pressed();

            if (!on_hover && data.is_hovered) {
                data.was_hovered = true;
            }

            if (data.is_hovered) {
                key.hovered = true;
                return data;
            }
        }

        return data;
    }

    bool Timeline::on_input(sInputData data)
    {
        IO::set_hover(this, data);

        if (data.was_hovered) {
            Engine::instance->vibrate_hand(HAND_RIGHT, HOVER_HAPTIC_AMPLITUDE, HOVER_HAPTIC_DURATION);
        }

        on_hover = true;

        return true;
    }

    float Timeline::x_to_time(float x)
    {
        return x * zoom * 0.01f;
    }

    float Timeline::time_to_x(float time)
    {
        return (time * (1.0f / zoom) * 100.0f);
    }

    void Timeline::clear()
    {
        keyframes.clear();

        selected_key_index = -1;
    }

    TimelineKeyframe* Timeline::get_selected_key()
    {
        if (selected_key_index != -1) {
            return &keyframes[selected_key_index];
        }

        return nullptr;
    }

    void Timeline::add_keyframe(float time, Keyframe* keyframe)
    {
        uint32_t index = keyframes.size();
        keyframes.push_back({ time, keyframe, index });
    }

    void Timeline::set_title(const std::string& new_title)
    {
        title->set_text(new_title);
    }

    void Timeline::on_insert_keyframe()
    {
        if (custom_on_insert_keyframe) {
            custom_on_insert_keyframe(this, 0u);
        }
    }

    void Timeline::on_edit_keyframe()
    {
        auto key = get_selected_key();
        if (key && custom_on_edit_keyframe) {
            custom_on_edit_keyframe(this, key->index);
        }
    }

    void Timeline::on_duplicate_keyframe()
    {
        auto key = get_selected_key();
        if (key && custom_on_duplicate_keyframe) {
            bool must_select_new = custom_on_duplicate_keyframe(this, key->index);
            if (must_select_new) {
                select_keyframe_by_index(keyframes.size() - 1u);
            }
        }
    }

    void Timeline::on_delete_keyframe()
    {
        auto key = get_selected_key();
        if (key && custom_on_delete_keyframe) {
            custom_on_delete_keyframe(this, key->index);
        }
    }

    void Timeline::on_close_timeline()
    {
        bool should_close = true;
        if (custom_on_close) {
            should_close = custom_on_close(this, 0u);
        }
        if (should_close) {
            set_visibility(false);
        }
    }
}
