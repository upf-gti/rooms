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
        : Panel2D(name, desc.position, { 0.0f, 0.0f }, ui::CREATE_3D), panel_size(desc.size), padding(desc.padding)
    {
        float inner_width = panel_size.x - padding * 2.0f;
        float inner_height = panel_size.y - padding * 2.0f;

        on_close = desc.close_fn;
        on_edit_keyframe = desc.edit_keyframe_fn;
        on_delete_keyframe = desc.delete_keyframe_fn;

        root = new ui::XRPanel(name + "_background", { 0.0f, 0.f }, panel_size, 0u, panel_color);
        add_child(root);

        ui::VContainer2D* column = new ui::VContainer2D(name + "_column", glm::vec2(padding));
        column->set_fixed_size({ inner_width, inner_height });
        root->add_child(column);

        // Title
        float title_text_scale = 22.0f;
        float title_y_corrected = desc.title_height * 0.5f - title_text_scale * 0.5f;
        ui::Container2D* title_container = new ui::Container2D(name + "_title", { 0.0f, 0.0f }, { inner_width - padding * 0.4f, desc.title_height });
        title = new ui::Text2D(desc.title.empty() ? "Inspector" : desc.title, { 0.0f, title_y_corrected }, title_text_scale, ui::TEXT_CENTERED | ui::SKIP_TEXT_RECT);
        time_text = new ui::Text2D("0.0", { padding * 2.0f + 64.f + 10.f, title_y_corrected + 8.0f }, 20.f, ui::SKIP_TEXT_RECT);
        auto edit_button = new ui::TextureButton2D("edit_timeline_keyframe", { "data/textures/edit.png", 0u, { padding * 2.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Edit" });
        auto delete_button = new ui::TextureButton2D("delete_timeline_keyframe", { "data/textures/delete.png", 0u, { padding * 2.0f + 32.f + 4.f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Delete" });
        close_button = new ui::TextureButton2D("close_timeline", { "data/textures/cross.png", 0u, { inner_width - padding * 4.0f, title_y_corrected }, glm::vec2(32.0f), colors::WHITE, "Close" });
        title_container->add_child(edit_button);
        title_container->add_child(delete_button);
        title_container->add_child(time_text);
        title_container->add_child(title);
        title_container->add_child(close_button);
        column->add_child(title_container);

        Node::bind("edit_timeline_keyframe", [&](const std::string& sg, void* data) {
            if (on_edit_keyframe) {
                on_edit_keyframe(this);
            }
        });

        Node::bind("delete_timeline_keyframe", [&](const std::string& sg, void* data) {
            bool deleted = false;
            if (on_delete_keyframe) {
                deleted = on_delete_keyframe(this);
            }
            if (deleted) {
                select_keyframe(nullptr);
            }
        });

        Node::bind("close_timeline", [&](const std::string& sg, void* data) {
            bool should_close = true;
            if (on_close) {
                should_close = on_close(this);
            }
            if (should_close) {
                set_visibility(false);
            }
        });

        render_background = false;

        // Body row
        body = new ui::Container2D(name + "_body", { 0.0f, 0.0f });
        body->set_fixed_size({ inner_width, panel_size.y - desc.title_height - padding * 3.0f });
        column->add_child(body);

        playhead = new ui::Panel2D(name + "_cursor", { inner_width * 0.5f,0.0f }, { 1.0f, panel_size.y - desc.title_height - padding * 3.0f }, 0u, Color(1.0f, 1.0f, 1.0f, 0.25f));
        playhead->set_priority(BUTTON_MARK);
        body->add_child(playhead);

        quad_surface = new Surface();
        quad_surface->create_quad(16.0f, 48.0f, true);

        frame_mesh = generate_keyframe_mesh(colors::GRAY);
        frame_mesh_hovered = generate_keyframe_mesh(colors::WHITE);
        frame_mesh_selected = generate_keyframe_mesh(Color(1.0f, 0.136f, 0.0f, 1.0f));
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

        if (Renderer::instance->get_openxr_available()) {
            material->set_is_2D(false);
        }

        MeshInstance* mesh_instance = new MeshInstance();
        mesh_instance->add_surface(quad_surface);
        mesh_instance->set_surface_material_override(mesh_instance->get_surface(0), material);

        return mesh_instance;
    }

    void Timeline::select_keyframe(TimelineKeyframe* key)
    {
        if (selected_key) {
            selected_key->selected = false;
        }

        if (key) {
            key->selected = true;
            current_time = key->time;
        }

        selected_key = key;
        
        time_dirty = true;
    }

    void Timeline::render()
    {
        if (!visibility)
            return;

        glm::vec2 scale = get_scale();
        glm::vec2 keyframe_size = { 16.0f, 48.0f };
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
            last_grab_position = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
        }

        if (Input::was_grab_released(HAND_RIGHT)) {
            grabbing = false;
        }

        root->set_priority(PANEL);

        auto renderer = Renderer::instance;

        if (renderer->get_openxr_available()) {

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
                glm::vec2 size = panel_size * 0.5f / glm::vec2(width, height);
                glm::vec3 new_pos = eye + forward * 0.35f;

                m = glm::translate(m, new_pos);
                m = m * glm::toMat4(get_rotation_to_face(new_pos, renderer->get_camera_eye(), { 0.0f, 1.0f, 0.0f }));
                m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });
                m = glm::translate(m, -glm::vec3(size, 0.0f));
                set_xr_transform(Transform::mat4_to_transform(m));

                root->set_priority(DRAGGABLE);
            }
        }

        // zooming
        {
            float w_delta = Input::get_mouse_wheel_delta();
            zoom -= w_delta * 0.10f;
            zoom = std::clamp(zoom, 0.3f, 3.0f);
        }

        time_dirty = false;

        const sInputData& keyframe_input_data = get_input_data();

        // ...

        const sInputData& root_input_data = root->get_input_data(true);

        // Horizontal scrolling
        if (root_input_data.is_hovered)
        {
            float scroll_dt = 0.0f;

            if (root_input_data.was_pressed) {
                IO::set_focus(root);
            }

            if (root_input_data.is_pressed && !IO::is_focus_type(Node2DClassType::HSLIDER)) {
                scroll_dt += (last_scroll_position.x - root_input_data.local_position.x);
            }

            if (root_input_data.was_released) {
                IO::set_focus(nullptr);
            }

            last_scroll_position = root_input_data.local_position;

            float dt_time = x_to_time(scroll_dt);

            float new_time = current_time + dt_time;
            new_time = std::clamp(new_time, 0.0f, keyframes.back().time);
            time_dirty |= (new_time != current_time);
            current_time = new_time;
        }

        std::string s = std::to_string(current_time);
        size_t idx = s.find('.') + 1;
        time_text->set_text(s.substr(0, idx + 2));

        Node2D::update(delta_time);
    }

    sInputData Timeline::get_input_data(bool ignore_focus)
    {
        glm::vec2 scale = get_scale();
        glm::vec2 keyframe_size = { 16.0f, 48.0f };
        glm::vec2 offset = glm::vec2(playhead->get_translation().x, body->get_translation().y + padding * scale.y);
        offset.x -= time_to_x(current_time) * scale.x;

        for (auto& key : keyframes) {
            glm::vec2 position = offset + glm::vec2(time_to_x(key.time), keyframe_size.y * 0.5f) * scale;

            if (position.x < (body->get_translation().x + padding * scale.x) || (position.x + keyframe_size.x * scale.x) >(body->get_translation().x + body->fixed_size.x * scale.x)) {
                continue;
            }

            sInputData data;

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
                    if (Renderer::instance->get_openxr_available()) {
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

            if (data.is_hovered) {
                key.hovered = true;
                return data;
            }
        }

        return {};
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
    }

    void Timeline::add_keyframe(float time, Keyframe* keyframe)
    {
        keyframes.push_back({ time, keyframe });
    }

    void Timeline::set_title(const std::string& new_title)
    {
        title->set_text(new_title);
    }
}
