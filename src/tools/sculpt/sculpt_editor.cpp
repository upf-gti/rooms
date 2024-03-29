#include "sculpt_editor.h"

#include "includes.h"

#include "framework/utils/utils.h"
#include "framework/nodes/ui.h"
#include "framework/input.h"
#include "framework/nodes/viewport_3d.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "engine/rooms_engine.h"

#include "spdlog/spdlog.h"
#include "imgui.h"
#include <glm/gtx/quaternion.hpp>

uint8_t SculptEditor::last_generated_material_uid = 0;

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    mirror_mesh = new MeshInstance3D();
    mirror_mesh->add_surface(RendererStorage::get_surface("quad"));
    mirror_mesh->scale(glm::vec3(0.25f));

    Material mirror_material;
    mirror_material.diffuse_texture = RendererStorage::get_texture("data/textures/mirror_quad_texture.png");
    mirror_material.shader = RendererStorage::get_shader("data/shaders/mesh_texture.wgsl", mirror_material);

    mirror_mesh->set_surface_material_override(mirror_mesh->get_surface(0), mirror_material);

    floor_grid_mesh = new MeshInstance3D();
    floor_grid_mesh->add_surface(RendererStorage::get_surface("quad"));
    floor_grid_mesh->set_translation(glm::vec3(0.0f));
    floor_grid_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    floor_grid_mesh->scale(glm::vec3(10.f));

    Material grid_material;
    grid_material.transparency_type = ALPHA_BLEND;
    grid_material.shader = RendererStorage::get_shader("data/shaders/mesh_grid.wgsl", grid_material);

    floor_grid_mesh->set_surface_material_override(mirror_mesh->get_surface(0), grid_material);

    axis_lock_gizmo.initialize(POSITION_GIZMO, sculpt_start_position);
    mirror_gizmo.initialize(POSITION_GIZMO, sculpt_start_position);
    mirror_origin = sculpt_start_position;

    // Initialize default primitive states
    {
        primitive_default_states[SD_SPHERE]     = { glm::vec4(0.02f, 0.0f, 0.0f, 0.0f) };
        primitive_default_states[SD_BOX]        = { glm::vec4(0.02f, 0.02f, 0.02f, 0.0f) };
        primitive_default_states[SD_CONE]       = { glm::vec4(0.05f, 0.0f, 0.0f, 0.03f) };
        primitive_default_states[SD_CYLINDER]   = { glm::vec4(0.05f, 0.0f, 0.0f, 0.03f) };
        primitive_default_states[SD_CAPSULE]    = { glm::vec4(0.05f, 0.0f, 0.0f, 0.03f) };
        primitive_default_states[SD_TORUS]      = { glm::vec4(0.03f, 0.0f, 0.0f, 0.01f) };
        primitive_default_states[SD_BEZIER]     = { glm::vec4(0.0) };
    }

    // Edit preview mesh
    {
        Surface* sphere_surface = new Surface();
        sphere_surface->create_sphere();

        mesh_preview = new MeshInstance3D();
        mesh_preview->add_surface(sphere_surface);

        Material preview_material;
        preview_material.priority = 1;
        preview_material.transparency_type = ALPHA_BLEND;
        preview_material.shader = RendererStorage::get_shader("data/shaders/mesh_transparent.wgsl", preview_material);

        mesh_preview->set_surface_material_override(sphere_surface, preview_material);

        mesh_preview_outline = new MeshInstance3D();
        mesh_preview_outline->add_surface(sphere_surface);

        Material outline_material;
        outline_material.cull_type = CULL_FRONT;
        outline_material.shader = RendererStorage::get_shader("data/shaders/mesh_outline.wgsl", outline_material);

        mesh_preview_outline->set_surface_material_override(sphere_surface, outline_material);
    }

    // Add pbr materials data
    {
        add_pbr_material_data("aluminium",   Color(0.912f, 0.914f, 0.92f, 1.0f),  0.0f, 1.0f);
        add_pbr_material_data("charcoal",    Color(0.02f, 0.02f, 0.02f, 1.0f),    0.5f, 0.0f);
        add_pbr_material_data("rusted_iron", Color(0.531f, 0.512f, 0.496f, 1.0f), 0.0f, 1.0f, 1.0f); // add noise
    }

    init_ui();

    if (Renderer::instance->get_openxr_available()) {
        main_panel_3d = new Viewport3D(main_panel_2d);
        RoomsEngine::entities.push_back(main_panel_3d);
    }

    // Load controller UI labels
    if (Renderer::instance->get_openxr_available())
    {
        // Left hand
        {
            left_hand_container = new ui::VContainer2D("left_controller_root", { 0.0f, 0.f });

            controller_labels[HAND_LEFT].secondary_button_label = new ui::ImageLabel2D("Change to Substract", "data/textures/buttons/y.png", 39.0f);
            left_hand_container->add_child(controller_labels[HAND_LEFT].secondary_button_label);
            /*controller_labels[HAND_LEFT].main_button_label = new ui::ImageLabel2D("Show UI", "data/textures/buttons/x.png", 30.0f);
            left_hand_container->add_child(controller_labels[HAND_LEFT].main_button_label);*/

            left_hand_ui_3D = new Viewport3D(left_hand_container);
            RoomsEngine::entities.push_back(left_hand_ui_3D);
        }

        // Right hand
        {
            right_hand_container = new ui::VContainer2D("right_controller_root", { 0.0f, 0.f });

            controller_labels[HAND_RIGHT].secondary_button_label = new ui::ImageLabel2D("Change to Stamp", "data/textures/buttons/b.png", 30.0f);
            right_hand_container->add_child(controller_labels[HAND_RIGHT].secondary_button_label);
            controller_labels[HAND_RIGHT].main_button_label = new ui::ImageLabel2D("Click on the UI", "data/textures/buttons/a.png", 30.0f);
            right_hand_container->add_child(controller_labels[HAND_RIGHT].main_button_label);

            right_hand_ui_3D = new Viewport3D(right_hand_container);
            RoomsEngine::entities.push_back(right_hand_ui_3D);
        }
    }

    // Load ui and Bind callbacks
    bind_events();

    enable_tool(SCULPT);

    renderer->change_stroke(stroke_parameters);
}

void SculptEditor::clean()
{
    if (mirror_mesh) {
        delete mirror_mesh;
    }

    if (floor_grid_mesh) {
        delete floor_grid_mesh;
    }

    // TODO
    // Clean all UI widgets
    // ...
}

bool SculptEditor::is_tool_being_used(bool stamp_enabled)
{
#ifdef XR_SUPPORT
    bool is_currently_pressed = Input::get_trigger_value(HAND_RIGHT) > 0.5f;
    is_released = is_tool_pressed && !is_currently_pressed;

    bool add_edit_with_tool = stamp_enabled ? is_released : Input::get_trigger_value(HAND_RIGHT) > 0.5f;

    // Update the is_pressed
    was_tool_pressed = is_tool_pressed;
    is_tool_pressed = is_currently_pressed;
    return Input::was_key_pressed(GLFW_KEY_SPACE) || add_edit_with_tool;
#else
    return Input::is_key_pressed(GLFW_KEY_SPACE);
#endif
}

bool SculptEditor::edit_update(float delta_time)
{
    // Poll action using stamp mode when picking material also mode to detect once
    bool is_tool_used = is_tool_being_used(stamp_enabled || is_picking_material);

    if (is_picking_material && is_tool_used)
    {
        pick_material();
        return false;
    }

    // Move the edit a little away
    glm::mat4x4 controller_pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
    controller_pose = glm::translate(controller_pose, glm::vec3(0.0f, 0.0f, -hand2edit_distance));

    // Update edit transform
    if ((!stamp_enabled || !is_tool_pressed) && !is_released) {
        edit_to_add.position = controller_pose[3];
        edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT, POSE_AIM));
        edit_position_stamp = edit_to_add.position;
        edit_origin_stamp = edit_to_add.position;
        edit_rotation_stamp = edit_to_add.rotation;
    }
    else {
        edit_to_add.position = edit_position_stamp;
        edit_to_add.rotation = edit_rotation_stamp;
    }

    // Snap surface
    if (canSnapToSurface()) {

        auto callback = [&](glm::vec3 center) {
            edit_to_add.position = texture3d_to_world(center);
        };

        glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        glm::vec3 ray_dir = get_front(pose);
        renderer->get_raymarching_renderer()->octree_ray_intersect(pose[3], ray_dir, callback);
    }

    // Update edit dimensions
    if (!stamp_enabled || !is_tool_pressed && !is_released) {
        // Get the data from the primitive default 
        edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;
        float size_multiplier = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 0.1f;
        dimensions_dirty |= (fabsf(size_multiplier) > 0.f);
        glm::vec3 new_dimensions = glm::clamp(size_multiplier + glm::vec3(edit_to_add.dimensions), 0.001f, 0.1f);
        edit_to_add.dimensions = glm::vec4(new_dimensions, edit_to_add.dimensions.w);

        // Update primitive specific size
        size_multiplier = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1f;
        edit_to_add.dimensions.w = glm::clamp(size_multiplier + edit_to_add.dimensions.w, 0.001f, 0.1f);
        dimensions_dirty |= (fabsf(size_multiplier) > 0.f);

        // Update in primitive state
        primitive_default_states[stroke_parameters.get_primitive()].dimensions = edit_to_add.dimensions;
        edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT, POSE_AIM));
        is_stretching_edit = false;
    }
    else if (stamp_enabled && is_tool_pressed) { // Stretch the edit using motion controls
        if (is_stretching_edit) {
            sdPrimitive curr_primitive = stroke_parameters.get_primitive();

            const glm::quat hand_rotation = (Input::get_controller_rotation(HAND_RIGHT, POSE_AIM));
            const glm::vec3 hand_position = controller_pose[3];

            const glm::vec3 stamp_origin_to_hand = edit_origin_stamp - hand_position;
            const float stamp_to_hand_distance = glm::length(stamp_origin_to_hand);
            const glm::vec3 stamp_to_hand_norm = stamp_origin_to_hand / (stamp_to_hand_distance);

            // Get rotation of the controller, along the stretch direction
            glm::quat swing, twist;
            quat_swing_twist_decomposition(stamp_to_hand_norm, hand_rotation, swing, twist);
            twist.w *= -1.0f;

            edit_rotation_stamp = get_quat_between_vec3(stamp_origin_to_hand, glm::vec3(0.0f, -stamp_to_hand_distance, 0.0f)) * twist;

            if (curr_primitive == SD_SPHERE) {
                edit_to_add.dimensions.x = stamp_to_hand_distance;
            }
            else if (curr_primitive == SD_CAPSULE) {
                edit_to_add.position = edit_origin_stamp;
                edit_to_add.dimensions.x = stamp_to_hand_distance;
            }
            else if (curr_primitive == SD_BOX) {
                edit_position_stamp = edit_origin_stamp + stamp_to_hand_norm * stamp_to_hand_distance * -0.5f;
                edit_to_add.dimensions.y = stamp_to_hand_distance * 0.5f;
            }
        }
        else {
            // Only stretch the edit when the acceleration of the hand exceds a threshold
            is_stretching_edit = glm::length(glm::abs(controller_velocity)) > 0.20f;
        }
        
    }

    // Edit modifiers
    {
        if (snap_to_grid) {
            float grid_multiplier = 1.f / snap_grid_size;
            // Uncomment for grid size of half of the edit radius
            // grid_multiplier = 1.f / (edit_to_add.dimensions.x / 2.f);
            edit_to_add.position = glm::round(edit_to_add.position * grid_multiplier) / grid_multiplier;
        }

        if (axis_lock) {

            is_tool_used &= !(axis_lock_gizmo.update(axis_lock_position, edit_to_add.position, delta_time));

            glm::vec3 locked_pos = edit_to_add.position;

            if (axis_lock_mode & AXIS_LOCK_X)
                locked_pos.x = axis_lock_position.x;
            else if (axis_lock_mode & AXIS_LOCK_Y)
                locked_pos.y = axis_lock_position.y;
            else if (axis_lock_mode & AXIS_LOCK_Z)
                locked_pos.z = axis_lock_position.z;

            edit_to_add.position = locked_pos;
            edit_to_add.rotation = glm::quat();
        }
    }

    // Update shape editor values
    {
        if (modifiers_dirty) {

            glm::vec4 parameters = stroke_parameters.get_parameters();
            parameters.x = onion_enabled ? onion_thickness : 0.0f;
            parameters.y = capped_enabled ? capped_value : 0.0f;
            stroke_parameters.set_parameters(parameters);

            modifiers_dirty = false;
        }
    }

    // Operation changer for the different tools
    {
        if (Input::was_button_pressed(XR_BUTTON_Y)) {
            sdOperation op = stroke_parameters.get_operation();
            std::string new_label_text = "";

            if (current_tool == SCULPT) {
                switch (op) {
                case OP_UNION:
                    op = OP_SUBSTRACTION;
                    new_label_text = "Change to Addition";
                    break;
                case OP_SUBSTRACTION:
                    op = OP_UNION;
                    new_label_text = "Change to Substraction";
                    break;
                case OP_SMOOTH_UNION:
                    op = OP_SMOOTH_SUBSTRACTION;
                    new_label_text = "Change to Smooth Addition";
                    break;
                case OP_SMOOTH_SUBSTRACTION:
                    op = OP_SMOOTH_UNION;
                    new_label_text = "Change to Smooth Substraction";
                    break;
                default:
                    new_label_text = "@Alex socorro";
                    break;
                }
            }
            else if (current_tool == PAINT) {
                op = OP_PAINT ? OP_SMOOTH_PAINT : OP_PAINT;
                new_label_text = (op == OP_PAINT) ? "Change to Smooth Paint" : "Change to Paint";
            }
            controller_labels[HAND_LEFT].secondary_button_label->set_text(new_label_text);
            stroke_parameters.set_operation(op);
        }
    }

    // Debug sculpting
    {
        if (Input::is_key_pressed(GLFW_KEY_P))
        {
            enable_tool(PAINT);
        }

        // For debugging sculpture without a headset
        if (!Renderer::instance->get_openxr_available()) {

            if (current_tool == SCULPT && is_tool_being_used(stamp_enabled)) {

                edit_to_add.position = glm::vec3(glm::vec3(0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1)));
                glm::vec3 euler_angles(random_f() * 90, random_f() * 90, random_f() * 90);
                edit_to_add.dimensions = glm::vec4(0.05f, 0.01f, 0.01f, 0.01f) * 1.0f;
                //edit_to_add.dimensions = (edit_to_add.operation == OP_SUBSTRACTION) ? 3.0f * glm::vec4(0.2f, 0.2f, 0.2f, 0.2f) : glm::vec4(0.2f, 0.2f, 0.2f, 0.2f);
                edit_to_add.rotation = glm::inverse(glm::quat(euler_angles));
                // Stroke
                //stroke_parameters.set_color(glm::vec4(0.1f, 0.1f, 0.1f, 1.f));
                //stroke_parameters.set_primitive((random_f() > 0.25f) ? ((random_f() > 0.5f) ? SD_SPHERE : SD_CYLINDER) : SD_BOX);
                ////stroke_parameters.primitive = (random_f() > 0.5f) ? SD_SPHERE : SD_BOX;
                //// stroke_parameters.material = glm::vec4(random_f(), random_f(), 0.f, 0.f);
                ////stroke_parameters.set_operation( (random_f() > 0.5f) ? OP_UNION : OP_SUBSTRACTION);
                //stroke_parameters.set_operation(OP_UNION);
                //stroke_parameters.set_material_metallic(0.9);
                //stroke_parameters.set_material_roughness(0.2);
                //stroke_parameters.set_smooth_factor(0.01);
            }
            else {
                // Make sure we don't get NaNs in preview rotation due to polling XR controllers in 2D mode
                edit_to_add.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
            }
        }
    }

    // Store now since later it will be converted to 3d texture space
    edit_position_world = edit_to_add.position;
    edit_rotation_world = edit_to_add.rotation;

    return is_tool_used;
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    // Update UI
    {
        if (main_panel_3d) {
            main_panel_3d->set_active(Input::is_button_pressed(XR_BUTTON_X));
            glm::mat4x4 pose = Input::get_controller_pose(HAND_LEFT, POSE_AIM);
            pose = glm::rotate(pose, glm::radians(-45.f), glm::vec3(1.0f, 0.0f, 0.0f));
            main_panel_3d->set_model(pose);
        }
        else {
            main_panel_2d->update(delta_time);

            if (is_picking_material && Input::was_mouse_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
                pick_material();
            }
        }

        // Update controller UI
        if (Renderer::instance->get_openxr_available())
        {
            glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT);
            pose = glm::rotate(pose, glm::radians(-110.f), glm::vec3(1.0f, 0.0f, 0.0f));
            right_hand_ui_3D->set_model(pose);

            pose = Input::get_controller_pose(HAND_LEFT);
            pose = glm::rotate(pose, glm::radians(-110.f), glm::vec3(1.0f, 0.0f, 0.0f));

            left_hand_ui_3D->set_model(pose);
        }
    }

    preview_tmp_edits.clear();
    new_edits.clear();

    // Update controller speed & acceleration
    {
        const glm::vec3 curr_controller_pos = Input::get_controller_position(HAND_RIGHT);
        const glm::vec3 curr_controller_velocity = (curr_controller_pos - prev_controller_pos) / delta_time;
        controller_acceleration = (curr_controller_velocity - controller_velocity) / delta_time;
        controller_velocity = curr_controller_velocity;
        prev_controller_pos = curr_controller_pos;

        //spdlog::info("{}", glm::length(glm::abs(controller_acceleration)));
    }

    bool is_tool_used = edit_update(delta_time);

    // Interaction input
    {
        if (Input::was_key_pressed(GLFW_KEY_U) || Input::was_grab_pressed(HAND_LEFT)) {
            renderer->undo();
        }

        if (Input::was_key_pressed(GLFW_KEY_R) || Input::was_grab_pressed(HAND_RIGHT)) {
            renderer->redo();
        }
    }

    // Sculpt lifecicle
    {
        // Set center of sculpture and reuse it as mirror center
        if (!sculpt_started) {
            sculpt_start_position = edit_to_add.position;
            renderer->set_sculpt_start_position(sculpt_start_position);
            mirror_origin = sculpt_start_position;
            axis_lock_position = sculpt_start_position;
        }

        // Mark the start of the sculpture for the origin
        if (current_tool == SCULPT && is_tool_used) {
            sculpt_started = true;
        }
    }

    scene_update_rotation();

    // Edit & Stroke submission
    {
        if (use_mirror) {
            is_tool_used &= !(mirror_gizmo.update(mirror_origin, mirror_rotation, edit_position_world, delta_time));
            mirror_normal = glm::normalize(mirror_rotation * glm::vec3(0.f, 0.f, 1.f));
        }
       

        // if any parameter changed or just stopped sculpting change the stroke
        if (stroke_parameters.is_dirty() || (was_tool_used && !is_tool_used)) {
            renderer->change_stroke(stroke_parameters);
            stroke_parameters.set_dirty(false);
        }

        // Upload the edit to the  edit list
        if (is_tool_used) {
            new_edits.push_back(edit_to_add);
            // Add recent color only when is used...
            add_recent_color(stroke_parameters.get_material().color);
            // Reset smear mode
            if (was_material_picked) {
                stamp_enabled = false;
            }
        }
    }

    // Set the edit as the preview
    preview_tmp_edits.push_back(edit_to_add);

    // Mirror functionality
    if (use_mirror) {
        mirror_current_edits(delta_time);
    }

    // Push to the renderer the edits and the previews
    renderer->push_preview_edit_list(preview_tmp_edits);
    renderer->push_edit_list(new_edits);

    was_tool_used = is_tool_used;
}

void SculptEditor::apply_mirror_position(glm::vec3& position)
{
    // Don't rotate the mirror origin..
    glm::vec3 origin_texture_space = world_to_texture3d(mirror_origin);
    glm::vec3 normal_texture_space = world_to_texture3d(mirror_normal, true);
    glm::vec3 pos_to_origin = origin_texture_space - position;
    glm::vec3 reflection = glm::reflect(pos_to_origin, normal_texture_space);
    position = origin_texture_space - reflection;
}

void SculptEditor::apply_mirror_rotation(glm::quat& rotation) const
{
    glm::vec3 curr_dir = rotation * glm::vec3{ 0.0f, 0.0f, 1.0f };
    glm::vec3 mirror_dir = glm::reflect(curr_dir, mirror_normal);

    rotation = get_quat_between_vec3(curr_dir, mirror_dir) * rotation;
}

void SculptEditor::mirror_current_edits(float delta_time)
{
    uint64_t preview_edit_count = preview_tmp_edits.size();

    for (uint64_t i = 0u; i < preview_edit_count; i++) {

        Edit mirrored_preview_edit = preview_tmp_edits[i];
        apply_mirror_position(mirrored_preview_edit.position);
        apply_mirror_rotation(mirrored_preview_edit.rotation);
        preview_tmp_edits.push_back(mirrored_preview_edit);
    }

    uint64_t edit_count = new_edits.size();

    for (uint64_t i = 0u; i < edit_count; i++) {

        Edit mirrored_edit = new_edits[i];
        apply_mirror_position(mirrored_edit.position);
        apply_mirror_rotation(mirrored_edit.rotation);
        new_edits.push_back(mirrored_edit);
    }
}

glm::vec3 SculptEditor::world_to_texture3d(const glm::vec3& position, bool skip_translation)
{
    glm::vec3 pos_texture_space = position;

    if (!skip_translation) {
        pos_texture_space -= (sculpt_start_position + translation_diff);
    }

    pos_texture_space = (sculpt_rotation * rotation_diff) * pos_texture_space;

    return pos_texture_space;
}

glm::vec3 SculptEditor::texture3d_to_world(const glm::vec3& position)
{
    glm::vec3 pos_world_space;

    pos_world_space = glm::inverse(sculpt_rotation * rotation_diff) * position;
    pos_world_space = pos_world_space + (sculpt_start_position + translation_diff);

    return pos_world_space;
}

void SculptEditor::scene_update_rotation()
{
    if (is_rotation_being_used()) {

        if (!rotation_started) {
            initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
            initial_hand_translation = Input::get_controller_position(HAND_LEFT);
        }

        rotation_diff = glm::inverse(initial_hand_rotation) * glm::inverse(Input::get_controller_rotation(HAND_LEFT));
        translation_diff = Input::get_controller_position(HAND_LEFT) - initial_hand_translation;

        renderer->set_sculpt_rotation(sculpt_rotation * rotation_diff);
        renderer->set_sculpt_start_position(sculpt_start_position + translation_diff);

        rotation_started = true;

        // Edit rotation WHILE rotation
        glm::quat tmp_rotation = sculpt_rotation * rotation_diff;
        edit_to_add.position -= (sculpt_start_position + translation_diff);
        edit_to_add.position = tmp_rotation * edit_to_add.position;
        edit_to_add.rotation *= (glm::conjugate(tmp_rotation));
    }
    else {
        // If rotation has stopped
        if (rotation_started) {
            sculpt_rotation = sculpt_rotation * rotation_diff;
            sculpt_start_position = sculpt_start_position + translation_diff;
            rotation_started = false;
            rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
            translation_diff = {};
        }

        // Push edits in 3d texture space
        edit_to_add.position = world_to_texture3d(edit_to_add.position);
        edit_to_add.rotation *= (glm::conjugate(sculpt_rotation) * rotation_diff);
    }

}

void SculptEditor::render()
{
    if (mesh_preview && renderer->get_openxr_available())
    {
        update_edit_preview(edit_to_add.dimensions);

        // Render something to be able to cull faces later...
        if (!mustRenderMeshPreviewOutline()) {
                mesh_preview->render();
        }
        else
        {
            mesh_preview_outline->set_model(mesh_preview->get_model());
            mesh_preview_outline->render();
        }
    }

    if (axis_lock) {
        axis_lock_gizmo.render();

        mirror_mesh->set_translation(axis_lock_position);
        if (axis_lock_mode & AXIS_LOCK_X)
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        else if (axis_lock_mode & AXIS_LOCK_Y)
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        // debug
        if (Input::was_key_pressed(GLFW_KEY_X))
            axis_lock_mode = AXIS_LOCK_X;
        if (Input::was_key_pressed(GLFW_KEY_Y))
            axis_lock_mode = AXIS_LOCK_Y;
        if (Input::was_key_pressed(GLFW_KEY_Z))
            axis_lock_mode = AXIS_LOCK_Z;

        mirror_mesh->render();
    }
    else if (use_mirror) {
        mirror_gizmo.render();
        mirror_mesh->set_translation(mirror_origin);
        mirror_mesh->scale(glm::vec3(0.25f));
        mirror_mesh->rotate(mirror_rotation);
        mirror_mesh->render();
    }

    floor_grid_mesh->render();

    if (!main_panel_3d) {
        main_panel_2d->render();
    }
}

// =====================
// GUI =================
// =====================

void SculptEditor::render_gui()
{
    StrokeMaterial& stroke_material = stroke_parameters.get_material();

    bool changed = false;

    ImGui::Text("Material");

    ImGui::Separator();

    ImGui::Text("PBR");
    changed |= ImGui::ColorEdit4("Base Color", &stroke_material.color[0]);
    changed |= ImGui::SliderFloat("Roughness", &stroke_material.roughness, 0.f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &stroke_material.metallic, 0.f, 1.0f);

    ImGui::Separator();

    ImGui::Text("Noise");
    changed |= ImGui::ColorEdit4("Color", &stroke_material.noise_color[0]);
    changed |= ImGui::SliderFloat("Intensity", &stroke_material.noise_params.x, 0.f, 1.0f);
    changed |= ImGui::SliderFloat("Frequency", &stroke_material.noise_params.y, 0.f, 50.0f);

    int tmp = static_cast<int>(stroke_material.noise_params.z);
    changed |= ImGui::SliderInt("Octaves", &tmp, 1, 16);
    stroke_material.noise_params.z = static_cast<float>(tmp);

    if (changed)
        stroke_parameters.set_dirty(true);
}

bool SculptEditor::canSnapToSurface()
{
    return snap_to_surface && (stamp_enabled || current_tool == PAINT);
}

bool SculptEditor::mustRenderMeshPreviewOutline()
{
    return stroke_parameters.get_operation() == OP_UNION || stroke_parameters.get_operation() == OP_SMOOTH_UNION;
}

void SculptEditor::update_edit_preview(const glm::vec4& dims)
{
    // Recreate mesh depending on primitive parameters

    if (dimensions_dirty)
    {
        // Expand a little bit the edges
        glm::vec4 grow_dims = dims;

        if (!mustRenderMeshPreviewOutline()) {
            grow_dims.x = std::max(grow_dims.x - 0.002f, 0.001f);
        }

        switch (stroke_parameters.get_primitive())
        {
        case SD_SPHERE:
            mesh_preview->get_surface(0)->create_sphere(grow_dims.x);
            break;
        case SD_BOX:
            if (dims.w > 0.001f) {
                mesh_preview->get_surface(0)->create_rounded_box(grow_dims.x, grow_dims.y, grow_dims.z, (dims.w / 0.1f) * grow_dims.x);
            }
            else {
                grow_dims += 0.002f;
                mesh_preview->get_surface(0)->create_box(grow_dims.x, grow_dims.y, grow_dims.z);
            }
            break;
        case SD_CONE:
            mesh_preview->get_surface(0)->create_cone(grow_dims.w, grow_dims.x);
            break;
        case SD_CYLINDER:
            mesh_preview->get_surface(0)->create_cylinder(grow_dims.w, grow_dims.x * 2.0f);
            break;
        case SD_CAPSULE:
            mesh_preview->get_surface(0)->create_capsule(grow_dims.w, grow_dims.x);
            break;
        case SD_TORUS:
            mesh_preview->get_surface(0)->create_torus(grow_dims.x, glm::clamp(grow_dims.w, 0.0001f, grow_dims.x));
            break;
        default:
            break;
        }

        spdlog::trace("Edit mesh preview generated!");

        dimensions_dirty = false;
    }

    glm::mat4x4 preview_pose = glm::translate(glm::mat4x4(1.0f), edit_position_world);
    preview_pose *= glm::inverse(glm::toMat4(edit_rotation_world));

    // Update edit transform
    mesh_preview->set_model(preview_pose);

    // Update model depending on the primitive
    switch (stroke_parameters.get_primitive())
    {
    case SD_CAPSULE:
        mesh_preview->translate({ 0.f, dims.x * 0.5, 0.f });
        break;
    default:
        break;
    }
}

void SculptEditor::set_sculpt_started(bool value)
{
    sculpt_started = true;
}

void SculptEditor::set_primitive(sdPrimitive primitive)
{
    stroke_parameters.set_primitive(primitive);
    dimensions_dirty = true;

    auto it = primitive_default_states.find(primitive);
    if (it != primitive_default_states.end())
    {
        edit_to_add.dimensions = (*it).second.dimensions;
    }
}

void SculptEditor::set_onion_modifier(float value)
{
    onion_thickness = glm::clamp(value, 0.0f, 1.0f);
    modifiers_dirty = true;
}

void SculptEditor::set_cap_modifier(float value)
{
    capped_value = glm::clamp(value, 0.0f, 1.0f);
    modifiers_dirty = true;
}

void SculptEditor::toggle_onion_modifier()
{
    //capped_enabled = false;
    onion_enabled = !onion_enabled;

    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.x = onion_enabled ? onion_thickness : 0.0f;

    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::toggle_capped_modifier()
{
    //onion_enabled = false;
    capped_enabled = !capped_enabled;

    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.y = capped_enabled ? capped_value : 0.0f;

    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::enable_tool(eTool tool)
{
    current_tool = tool;

    switch (tool)
    {
    case SCULPT:
        // helper_gui.change_list_layout("sculpt");
        stroke_parameters.set_operation(OP_UNION);
        hand2edit_distance = 0.0f;
        break;
    case PAINT:
        // helper_gui.change_list_layout("paint");
        stroke_parameters.set_operation(OP_PAINT);
        hand2edit_distance = 0.1f;
        break;
    default:
        break;
    }

    // Set this to allow the mesh preview to give a little mergin in the outline mode
    dimensions_dirty = true;
}

bool SculptEditor::is_rotation_being_used()
{
    return Input::get_trigger_value(HAND_LEFT) > 0.5;
}

void SculptEditor::init_ui()
{
    main_panel_2d = new ui::HContainer2D("root", { 12.0f, 400.f });

    {
        {
            ui::ItemGroup2D* g_main_tools = new ui::ItemGroup2D("g_main_tools");
            g_main_tools->add_child(new ui::TextureButton2D("sculpt", "data/textures/cube.png", ui::UNIQUE_SELECTION | ui::SELECTED));
            g_main_tools->add_child(new ui::TextureButton2D("paint", "data/textures/paint.png", ui::UNIQUE_SELECTION));
            main_panel_2d->add_child(g_main_tools);
        }

        {
            ui::ButtonSelector2D* primitives_submenu = new ui::ButtonSelector2D("primitives", "data/textures/primitives.png");
            primitives_submenu->add_child(new ui::TextureButton2D("sphere", "data/textures/sphere.png", ui::UNIQUE_SELECTION | ui::SELECTED));
            primitives_submenu->add_child(new ui::TextureButton2D("cube", "data/textures/cube.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(new ui::TextureButton2D("cone", "data/textures/cone.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(new ui::TextureButton2D("capsule", "data/textures/capsule.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(new ui::TextureButton2D("cylinder", "data/textures/cylinder.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(new ui::TextureButton2D("torus", "data/textures/torus.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(new ui::TextureButton2D("bezier", "data/textures/bezier.png", ui::UNIQUE_SELECTION));
            main_panel_2d->add_child(primitives_submenu);
        }

        {
            ui::ButtonSubmenu2D* material_submenu = new ui::ButtonSubmenu2D("material", "data/textures/material.png");

            {
                ui::ButtonSubmenu2D* colors_submenu = new ui::ButtonSubmenu2D("colors", "data/textures/colors.png");

                {
                    ui::ItemGroup2D* g_picker = new ui::ItemGroup2D("g_picker");
                    g_picker->add_child(new ui::ColorPicker2D("color_picker", colors::RED));
                    colors_submenu->add_child(g_picker);
                }

                {
                    ui::ItemGroup2D* g_colors = new ui::ItemGroup2D("g_colors");

                    // Color palette 1
                    {
                        ui::ButtonSelector2D* palette_1_submenu = new ui::ButtonSelector2D("color_template_palette_1", "data/textures/colors_template_1.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.2f, 0.21f, 0.77f, 1.0f), Color(0.41f, 0.57f, 0.79f, 1.0f), Color(0.41f, 0.76f, 0.79f, 1.0f), Color(0.64f, 0.9f, 0.93f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_1_submenu->add_child(new ui::Button2D("colors_t1_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_1_submenu);
                    }

                    // Color palette 2
                    {
                        ui::ButtonSelector2D* palette_2_submenu = new ui::ButtonSelector2D("color_template_palette_2", "data/textures/colors_template_2.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.36f, 0.61f, 0.35f, 1.0f), Color(0.51f, 0.8f, 0.49f, 1.0f), Color(0.68f, 0.84f, 0.05f, 1.0f), Color(0.71f, 0.96f, 0.49f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_2_submenu->add_child(new ui::Button2D("colors_t2_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_2_submenu);
                    }

                    // Color palette 3
                    {
                        ui::ButtonSelector2D* palette_3_submenu = new ui::ButtonSelector2D("color_template_palette_3", "data/textures/colors_template_3.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.94f, 0.77f, 0.44f, 1.0f), Color(0.96f, 0.67f, 0.09f, 1.0f), Color(0.98f, 0.9f, 0.08f, 1.0f), Color(0.94f, 0.92f, 0.7f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_3_submenu->add_child(new ui::Button2D("colors_t3_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_3_submenu);
                    }

                    // Color palette 4
                    {
                        ui::ButtonSelector2D* palette_4_submenu = new ui::ButtonSelector2D("color_template_palette_4", "data/textures/colors_template_4.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.58f, 0.12f, 0.12f, 1.0f), Color(0.83f, 0.09f, 0.09f, 1.0f), Color(0.88f, 0.42f, 0.42f, 1.0f), Color(0.90f, 0.71f, 0.71f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_4_submenu->add_child(new ui::Button2D("colors_t4_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_4_submenu);
                    }

                    // Color palette 5
                    {
                        ui::ButtonSelector2D* palette_5_submenu = new ui::ButtonSelector2D("color_template_palette_5", "data/textures/colors_template_5.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.65f, 0.02f, 0.83f, 1.0f), Color(0.8f, 0.49f, 0.89f, 1.0f), Color(0.95f, 0.12f, 0.85f, 1.0f), Color(0.95f, 0.62f, 0.91f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_5_submenu->add_child(new ui::Button2D("colors_t5_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_5_submenu);
                    }

                    // Color palette 6
                    {
                        ui::ButtonSelector2D* palette_6_submenu = new ui::ButtonSelector2D("color_template_palette_6", "data/textures/colors_template_6.png", ui::KEEP_RGB);
                        Color colors[4] = { Color(0.f, 0.f, 0.f, 1.0f), Color(0.33f, 0.33f, 0.33f, 1.0f), Color(0.66f, 0.66f, 0.66f, 1.0f), Color(1.0f, 1.0f, 1.0f, 1.0f) };
                        for (size_t i = 0; i < 4; i++) {
                            palette_6_submenu->add_child(new ui::Button2D("colors_t6_" + std::to_string(i), colors[i], ui::KEEP_RGB));
                        }
                        g_colors->add_child(palette_6_submenu);
                    }

                    {
                        ui::ButtonSelector2D* recent_colors_submenu = new ui::ButtonSelector2D("recent_colors", "data/textures/recent_colors.png", ui::KEEP_RGB);
                        for (size_t i = 0; i < 5; i++) {
                            recent_colors_submenu->add_child(new ui::Button2D("recent_color_" + std::to_string(i), colors::WHITE, ui::KEEP_RGB));
                        }
                        g_colors->add_child(recent_colors_submenu);
                    }

                    colors_submenu->add_child(g_colors);
                }

                material_submenu->add_child(colors_submenu);
            }

            {
                ui::ButtonSelector2D* color_blending_selector = new ui::ButtonSelector2D("color_blending", "data/textures/x.png");
                color_blending_selector->add_child(new ui::TextureButton2D("replace", "data/textures/y.png", ui::UNIQUE_SELECTION));
                color_blending_selector->add_child(new ui::TextureButton2D("mix", "data/textures/y.png", ui::UNIQUE_SELECTION));
                color_blending_selector->add_child(new ui::TextureButton2D("additive", "data/textures/y.png", ui::UNIQUE_SELECTION));
                color_blending_selector->add_child(new ui::TextureButton2D("multiply", "data/textures/y.png", ui::UNIQUE_SELECTION));
                color_blending_selector->add_child(new ui::TextureButton2D("screen", "data/textures/y.png", ui::UNIQUE_SELECTION));
                material_submenu->add_child(color_blending_selector);
            }

            {
                ui::ButtonSubmenu2D* material_editor_submenu = new ui::ButtonSubmenu2D("material_editor", "data/textures/material_editor.png");

                {
                    ui::ItemGroup2D* g_edit_pbr = new ui::ItemGroup2D("g_edit_pbr");
                    g_edit_pbr->add_child(new ui::Slider2D("roughness", 0.7f));
                    g_edit_pbr->add_child(new ui::Slider2D("metallic", 0.2f));
                    material_editor_submenu->add_child(g_edit_pbr);
                }

                {
                    ui::ItemGroup2D* g_edit_pattern = new ui::ItemGroup2D("g_edit_pattern");
                    g_edit_pattern->add_child(new ui::Slider2D("noise_intensity", 0.0f, ui::SliderMode::VERTICAL, 0.0f, 10.0f));
                    g_edit_pattern->add_child(new ui::Slider2D("noise_frequency", 20.0f, ui::SliderMode::VERTICAL, 0.0f, 50.0f));
                    g_edit_pattern->add_child(new ui::Slider2D("noise_octaves", 8.0f, ui::SliderMode::VERTICAL, 0.0f, 16.0f, 1.0f));
                    g_edit_pattern->add_child(new ui::ColorPicker2D("noise_color_picker", colors::WHITE));
                    material_editor_submenu->add_child(g_edit_pattern);
                }

                material_submenu->add_child(material_editor_submenu);
            }

            {
                ui::ButtonSubmenu2D* mat_list_submenu = new ui::ButtonSubmenu2D("materials", "data/textures/material_samples.png");

                mat_list_submenu->add_child(new ui::TextureButton2D("save_material", "data/textures/submenu_mark.png"));
                mat_list_submenu->add_child(new ui::TextureButton2D("pick_material", "data/textures/pick_material.png", ui::ALLOW_TOGGLE));

                {
                    ui::ButtonSelector2D* g_material_samples = new ui::ButtonSelector2D("g_material_samples", "data/textures/material_samples.png");
                    g_material_samples->add_child(new ui::TextureButton2D("aluminium", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                    g_material_samples->add_child(new ui::TextureButton2D("charcoal", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                    g_material_samples->add_child(new ui::TextureButton2D("rusted_iron", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                    mat_list_submenu->add_child(g_material_samples);
                }

                material_submenu->add_child(mat_list_submenu);
            }

            {
                material_submenu->add_child(new ui::TextureButton2D("shuffle_material", "data/textures/shuffle.png"));
            }

            main_panel_2d->add_child(material_submenu);
        }

        {
            ui::ItemGroup2D* g_utilities = new ui::ItemGroup2D("g_utilities");

            {
                ui::ButtonSubmenu2D* shape_editor_submenu = new ui::ButtonSubmenu2D("shape_editor", "data/textures/shape_editor.png");

                {
                    ui::ItemGroup2D* g_onion = new ui::ItemGroup2D("g_onion");
                    g_onion->add_child(new ui::TextureButton2D("onion", "data/textures/onion.png", ui::ALLOW_TOGGLE));
                    g_onion->add_child(new ui::Slider2D("onion_value", 0.0f, ui::SliderMode::VERTICAL, 0.01f));
                    shape_editor_submenu->add_child(g_onion);
                }

                {
                    ui::ItemGroup2D* g_capped = new ui::ItemGroup2D("g_capped");
                    g_capped->add_child(new ui::TextureButton2D("capped", "data/textures/capped.png", ui::ALLOW_TOGGLE));
                    g_capped->add_child(new ui::Slider2D("cap_value", 0.0f, ui::SliderMode::VERTICAL));
                    shape_editor_submenu->add_child(g_capped);
                }

                g_utilities->add_child(shape_editor_submenu);
            }

            {
                ui::ButtonSubmenu2D* mirror_submenu = new ui::ButtonSubmenu2D("mirror", "data/textures/mirror.png");
                mirror_submenu->add_child(new ui::TextureButton2D("mirror_toggle", "data/textures/mirror.png", ui::ALLOW_TOGGLE));
                ui::ComboButtons2D* g_mirror = new ui::ComboButtons2D("g_mirror");
                g_mirror->add_child(new ui::TextureButton2D("mirror_translation", "data/textures/translation_gizmo.png", ui::SELECTED));
                g_mirror->add_child(new ui::TextureButton2D("mirror_rotation", "data/textures/rotation_gizmo.png"));
                g_mirror->add_child(new ui::TextureButton2D("mirror_both", "data/textures/transform_gizmo.png"));
                mirror_submenu->add_child(g_mirror);
                g_utilities->add_child(mirror_submenu);
            }

            g_utilities->add_child(new ui::TextureButton2D("snap_to_surface", "data/textures/snap_to_surface.png", ui::ALLOW_TOGGLE));
            g_utilities->add_child(new ui::TextureButton2D("snap_to_grid", "data/textures/snap_to_grid.png", ui::ALLOW_TOGGLE));

            {
                ui::ButtonSubmenu2D* lock_axis_submenu = new ui::ButtonSubmenu2D("lock_axis", "data/textures/lock_axis.png");
                lock_axis_submenu->add_child(new ui::TextureButton2D("lock_axis_toggle", "data/textures/lock_axis.png", ui::ALLOW_TOGGLE));
                ui::ComboButtons2D* g_lock_axis = new ui::ComboButtons2D("g_lock_axis");
                g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_x", "data/textures/x.png"));
                g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_y", "data/textures/y.png"));
                g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_z", "data/textures/z.png", ui::SELECTED));
                lock_axis_submenu->add_child(g_lock_axis);
                g_utilities->add_child(lock_axis_submenu);
            }

            main_panel_2d->add_child(g_utilities);
        }
    }
}

void SculptEditor::bind_events()
{
    Node::bind("sculpt", [&](const std::string& signal, void* button) { enable_tool(SCULPT); });
    Node::bind("paint", [&](const std::string& signal, void* button) { enable_tool(PAINT); });

    Node::bind("sphere", [&](const std::string& signal, void* button) {  set_primitive(SD_SPHERE); });
    Node::bind("cube", [&](const std::string& signal, void* button) { set_primitive(SD_BOX); });
    Node::bind("cone", [&](const std::string& signal, void* button) { set_primitive(SD_CONE); });
    Node::bind("capsule", [&](const std::string& signal, void* button) { set_primitive(SD_CAPSULE); });
    Node::bind("cylinder", [&](const std::string& signal, void* button) { set_primitive(SD_CYLINDER); });
    Node::bind("torus", [&](const std::string& signal, void* button) { set_primitive(SD_TORUS); });
    Node::bind("bezier", [&](const std::string& signal, void* button) { set_primitive(SD_BEZIER); });

    Node::bind("onion", [&](const std::string& signal, void* button) { toggle_onion_modifier(); });
    Node::bind("onion_value", [&](const std::string& signal, float value) { set_onion_modifier(value); });
    Node::bind("capped", [&](const std::string& signal, void* button) { toggle_capped_modifier(); });
    Node::bind("cap_value", [&](const std::string& signal, float value) { set_cap_modifier(value); });

    Node::bind("mirror_toggle", [&](const std::string& signal, void* button) { use_mirror = !use_mirror; });
    Node::bind("mirror_translation", [&](const std::string& signal, void* button) { mirror_gizmo.set_mode(eGizmoType::POSITION_GIZMO); });
    Node::bind("mirror_rotation", [&](const std::string& signal, void* button) { mirror_gizmo.set_mode(eGizmoType::ROTATION_GIZMO); });
    Node::bind("mirror_both", [&](const std::string& signal, void* button) { mirror_gizmo.set_mode(eGizmoType::POSITION_ROTATION_GIZMO); });
    Node::bind("snap_to_surface", [&](const std::string& signal, void* button) { snap_to_surface = !snap_to_surface; });
    Node::bind("snap_to_grid", [&](const std::string& signal, void* button) { snap_to_grid = !snap_to_grid; });
    Node::bind("lock_axis_toggle", [&](const std::string& signal, void* button) { axis_lock = !axis_lock; });
    Node::bind("lock_axis_x", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_X; });
    Node::bind("lock_axis_y", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Y; });
    Node::bind("lock_axis_z", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Z; });

    Node::bind("roughness", [&](const std::string& signal, float value) { stroke_parameters.set_material_roughness(value); });
    Node::bind("metallic", [&](const std::string& signal, float value) { stroke_parameters.set_material_metallic(value); });
    Node::bind("noise_intensity", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(value); });
    Node::bind("noise_frequency", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(-1.0f, value); });
    Node::bind("noise_octaves", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(-1.0f, -1.0f, static_cast<int>(value)); });
    Node::bind("noise_color_picker", [&](const std::string& signal, Color color) { stroke_parameters.set_material_noise_color(color); });

    Node::bind("color_picker", [&](const std::string& signal, Color color) { stroke_parameters.set_material_color(color); });
    Node::bind("pick_material", [&](const std::string& signal, void* button) { is_picking_material = !is_picking_material; });

    // Bind colors callback...

    for (auto it : Node2D::all_widgets)
    {
        ui::Button2D* child = dynamic_cast<ui::Button2D*>(it.second);
        if (!child || !child->is_color_button) continue;
        Node::bind(child->signal, [&](const std::string& signal, void* button) {
            const Color& color = (reinterpret_cast<ui::Button2D*>(button))->color;
            stroke_parameters.set_material_color(color);
        });
    }

    // Bind recent color buttons...

    Node2D* recent_group = Node2D::get_widget_from_name("recent_colors");
    if (recent_group) {
        max_recent_colors = recent_group->get_children().size();
        for (size_t i = 0; i < max_recent_colors; ++i)
        {
            ui::Button2D* child = static_cast<ui::Button2D*>(recent_group->get_children()[i]);
            Node::bind(child->signal, [&](const std::string& signal, void* button) {
                const Color& color = (reinterpret_cast<ui::Button2D*>(button))->color;
                stroke_parameters.set_material_color(color);
            });
        }
    }
    else {
        spdlog::error("Cannot find recent_colors button group!");
    }

    // Bind material samples callback...

    Node2D* samples_group = Node2D::get_widget_from_name("g_material_samples");
    if (samples_group) {
        for (size_t i = 0; i < samples_group->get_children().size(); ++i)
        {
            ui::Button2D* child = static_cast<ui::Button2D*>(samples_group->get_children()[i]);
            Node::bind(child->signal, [&](const std::string& signal, void* button) {
                update_stroke_from_material(signal);
            });
        }
    }
    else {
        spdlog::error("Cannot find material_samples button group!");
    }

    Node::bind("save_material", [&](const std::string& signal, void* button) {
        generate_material_from_stroke(button);
    });

    Node::bind("shuffle_material", [&](const std::string& signal, void* button) {
        generate_random_material();
    });

    // Bind color blendind operations for painting
    {
        Node2D* color_blending_modes = Node2D::get_widget_from_name("color_blending");
        if (color_blending_modes) {
            for (size_t i = 0; i < color_blending_modes->get_children().size(); ++i)
            {
                ui::Button2D* child = static_cast<ui::Button2D*>(color_blending_modes->get_children()[i]);
                Node::bind(child->signal, [&, index = i](const std::string& signal, void* button) {
                    stroke_parameters.set_color_blend_operation(static_cast<ColorBlendingOp>(index));
                });
            }
        }
        else {
            spdlog::error("Cannot find color_blending_modes selector!");
        }
    }

    // Bind Controller buttons
    {
        Node::bind(XR_BUTTON_B, [&]() {
            stamp_enabled = !stamp_enabled;
            controller_labels[HAND_RIGHT].secondary_button_label->set_text(stamp_enabled ? "Switch to Smear" : "Switch to Stamp");
        });
    }
}

void SculptEditor::add_recent_color(const Color& color)
{
    auto it = std::find(recent_colors.begin(), recent_colors.end(), color);

    // Color is already in recents...
    if (it != recent_colors.end())
    {
        recent_colors.erase(it);
    }

    // Always add at the beginning
    recent_colors.insert(recent_colors.begin(), color);

    if (recent_colors.size() > max_recent_colors)
    {
        recent_colors.pop_back();
    }

    Node2D* recent_group = Node2D::get_widget_from_name("recent_colors");
    if (!recent_group) {
        return;
    }

    assert(recent_colors.size() <= recent_group->get_children().size());
    for (uint8_t i = 0; i < recent_colors.size(); ++i)
    {
        ui::Button2D* child = static_cast<ui::Button2D*>(recent_group->get_children()[i]);
        child->set_color(recent_colors[i]);
    }
}

void SculptEditor::add_pbr_material_data(const std::string& name, const Color& base_color, float roughness, float metallic,
    float noise_intensity, const Color& noise_color, float noise_frequency, int noise_octaves)
{
    pbr_materials_data[name] = {
        .base_color = base_color,
        .roughness = roughness,
        .metallic = metallic,
        .noise_params = glm::vec4(noise_intensity, noise_frequency, static_cast<float>(noise_octaves), 0.0f),
        .noise_color = noise_color
    };
}

void SculptEditor::generate_material_from_stroke(void* button)
{
    // Max of 5 materials
    if (num_generated_materials == 5) {
        return;
    }

    ui::Button2D* b = reinterpret_cast<ui::Button2D*>(button);
    ui::ButtonSelector2D* mat_samples = static_cast<ui::ButtonSelector2D*>(Node2D::get_widget_from_name("g_material_samples"));
    assert(mat_samples);

    std::string name = "new_material_" + std::to_string(last_generated_material_uid++);
    ui::TextureButton2D* new_button = new ui::TextureButton2D(name, "data/textures/material_samples.png", ui::UNIQUE_SELECTION);
    mat_samples->add_child(new_button);

    // Set button as 3d
    if (main_panel_3d) {
        new_button->remove_flag(MATERIAL_2D);
    }

    num_generated_materials++;

    // Add data to existing samples..
    const StrokeMaterial& mat = stroke_parameters.get_material();
    add_pbr_material_data(name, mat.color, mat.roughness, mat.metallic,
        mat.noise_params.x, mat.noise_color, mat.noise_params.y, static_cast<int>(mat.noise_params.z));

    Node::bind(name, [&](const std::string& signal, void* button) {
        update_stroke_from_material(signal);
    });
}

void SculptEditor::generate_random_material()
{
    // Set all data
    stroke_parameters.set_material_color(Color(random_f(), random_f(), random_f(), 1.0f));
    stroke_parameters.set_material_roughness(random_f());
    stroke_parameters.set_material_metallic(random_f());

    // Don't apply noise by now..
    stroke_parameters.set_material_noise();

    update_gui_from_stroke_material(stroke_parameters.get_material());
}

void SculptEditor::update_gui_from_stroke_material(const StrokeMaterial& mat)
{
    // Emit signals to change UI values
    Node::emit_signal("color_picker@changed", mat.color);
    Node::emit_signal("roughness@changed", mat.roughness);
    Node::emit_signal("metallic@changed", mat.metallic);
    Node::emit_signal("noise_intensity@changed", mat.noise_params.x);
    Node::emit_signal("noise_frequency@changed", mat.noise_params.y);
    Node::emit_signal("noise_octaves@changed", mat.noise_params.z);
    Node::emit_signal("noise_color_picker@changed", mat.noise_color);
}

void SculptEditor::update_stroke_from_material(const std::string& name)
{
    const PBRMaterialData& data = pbr_materials_data[name];

    // Set all data
    stroke_parameters.set_material_color(data.base_color);
    stroke_parameters.set_material_roughness(data.roughness * 1.5f); // this is a hack because hdres don't have too much roughness..
    stroke_parameters.set_material_metallic(data.metallic);
    stroke_parameters.set_material_noise(data.noise_params.x, data.noise_params.y, static_cast<int>(data.noise_params.z));

    update_gui_from_stroke_material(stroke_parameters.get_material());
}

void SculptEditor::pick_material()
{
    if (Renderer::instance->get_openxr_available())
    {
        glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        glm::vec3 ray_dir = get_front(pose);
        renderer->get_raymarching_renderer()->octree_ray_intersect(pose[3], ray_dir);
    }else
    {
        WebGPUContext* webgpu_context = Renderer::instance->get_webgpu_context();
        Camera* camera = Renderer::instance->get_camera();
        const glm::mat4x4& view_projection_inv = glm::inverse(camera->get_view_projection());

        glm::vec2 mouse_pos = Input::get_mouse_position();
        glm::vec3 mouse_pos_ndc;
        mouse_pos_ndc.x = (mouse_pos.x / webgpu_context->render_width) * 2.0f - 1.0f;
        mouse_pos_ndc.y = -((mouse_pos.y / webgpu_context->render_height) * 2.0f - 1.0f);
        mouse_pos_ndc.z = 1.0f;

        glm::vec4 ray_dir = view_projection_inv * glm::vec4(mouse_pos_ndc, 1.0f);
        ray_dir /= ray_dir.w;

        renderer->get_raymarching_renderer()->octree_ray_intersect(camera->get_eye(), glm::normalize(glm::vec3(ray_dir)));
    }

    const RayIntersectionInfo& info = static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->get_ray_intersection_info();

    if (info.intersected)
    {
        // Set all data
        stroke_parameters.set_material_color(Color(info.material_albedo, 1.0f));
        stroke_parameters.set_material_roughness(info.material_roughness);
        stroke_parameters.set_material_metallic(info.material_metalness);
        // stroke_parameters.set_material_noise(-1.0f);

        update_gui_from_stroke_material(stroke_parameters.get_material());
    }

    // Disable picking..
    Node::emit_signal("pick_material", (void*)nullptr);

    // Manage interactions, set stamp mode until tool is used again
    stamp_enabled = true;
    was_material_picked = true;
}
