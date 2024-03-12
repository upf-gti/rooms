#include "sculpt_editor.h"

#include "includes.h"

#include "framework/utils/utils.h"
#include "framework/nodes/ui.h"
#include "framework/input.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

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
    mirror_material.flags |= MATERIAL_DIFFUSE;
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
        primitive_default_states[SD_SPHERE]     = { glm::vec4(0.0) };
        primitive_default_states[SD_BOX]        = { glm::vec4(0.0) };
        primitive_default_states[SD_CONE]       = { glm::vec4(0.0) };
        primitive_default_states[SD_CYLINDER]   = { glm::vec4(0.0) };
        primitive_default_states[SD_CAPSULE]    = { glm::vec4(0.0) };
        primitive_default_states[SD_TORUS]      = { glm::vec4(0.0) };
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
        preview_material.cull_type = CULL_FRONT;
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

            edit_rotation_stamp = get_quat_between_vec3(stamp_origin_to_hand, glm::vec3(0.0f, 0.0f, stamp_to_hand_distance)) * twist;

            if (curr_primitive == SD_SPHERE) {
                edit_to_add.dimensions.x = stamp_to_hand_distance;
            }
            else if (curr_primitive == SD_CAPSULE) {
                edit_to_add.position = edit_origin_stamp;
                edit_to_add.dimensions.x = stamp_to_hand_distance;
            }
            else if (curr_primitive == SD_BOX) {
                edit_position_stamp = edit_origin_stamp + stamp_to_hand_norm * stamp_to_hand_distance * -0.5f;
                edit_to_add.dimensions.z = stamp_to_hand_distance * 0.5f;
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

            if (current_tool == SCULPT) {
                switch (op) {
                case OP_UNION:
                    op = OP_SUBSTRACTION;
                    break;
                case OP_SUBSTRACTION:
                    op = OP_UNION;
                    break;
                case OP_SMOOTH_UNION:
                    op = OP_SMOOTH_SUBSTRACTION;
                    break;
                case OP_SMOOTH_SUBSTRACTION:
                    op = OP_SMOOTH_UNION;
                    break;
                default:
                    break;
                }
            }
            else if (current_tool == PAINT) {
                op = OP_PAINT ? OP_SMOOTH_PAINT : OP_PAINT;
            }

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
        is_tool_used &= !(mirror_gizmo.update(mirror_origin, mirror_rotation, edit_position_world, delta_time));

        mirror_normal = glm::normalize(mirror_rotation * glm::vec3(0.f, 0.f, 1.f));

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
            mesh_preview->get_surface(0)->create_rounded_box(grow_dims.x, grow_dims.y, grow_dims.z, (dims.w / 0.1f) * grow_dims.x);
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
    case SD_CONE:
        mesh_preview->rotate(glm::radians(-90.f), { 1.f, 0.f, 0.f });
        break;
    case SD_CAPSULE:
        mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
        mesh_preview->translate({ 0.f, -dims.x * 0.5, 0.f });
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
    capped_enabled = false;
    onion_enabled = !onion_enabled;

    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.x = onion_enabled ? onion_thickness : 0.0f;

    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::toggle_capped_modifier()
{
    onion_enabled = false;
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

void SculptEditor::bind_events()
{
    // Set events
    {
        Node::bind("sculpt", [&](const std::string& signal, void* button) { enable_tool(SCULPT); });
        Node::bind("paint", [&](const std::string& signal, void* button) { enable_tool(PAINT); });

        Node::bind("sphere", [&](const std::string& signal, void* button) {  set_primitive(SD_SPHERE); });
        Node::bind("cube", [&](const std::string& signal, void* button) { set_primitive(SD_BOX); });
        Node::bind("cone", [&](const std::string& signal, void* button) { set_primitive(SD_CONE); });
        Node::bind("capsule", [&](const std::string& signal, void* button) { set_primitive(SD_CAPSULE); });
        Node::bind("cylinder", [&](const std::string& signal, void* button) { set_primitive(SD_CYLINDER); });
        Node::bind("torus", [&](const std::string& signal, void* button) { set_primitive(SD_TORUS); });

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
            if (it.second->get_class_type() != Node2DClassType::BUTTON) continue;
            ui::Button2D* child = static_cast<ui::Button2D*>(it.second);
            if (child->is_color_button) {
                Node::bind(child->signal, [&](const std::string& signal, void* button) {
                    const Color& color = (reinterpret_cast<ui::Button2D*>(button))->color;
                    stroke_parameters.set_material_color(color);
                });
            }
        }

        // Bind recent color buttons...

        Node2D* recent_group = Node2D::get_widget_from_name("g_recent_colors");
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
    }

    // Bind Controller buttons
    {
        Node::bind(XR_BUTTON_B, [&]() { stamp_enabled = !stamp_enabled; });
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

    Node2D* recent_group = Node2D::get_widget_from_name("g_recent_colors");
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
    ui::ItemGroup2D* mat_samples = static_cast<ui::ItemGroup2D*>(Node2D::get_widget_from_name("g_material_samples"));
    assert(mat_samples);

    std::string name = "new_material_" + std::to_string(last_generated_material_uid++);
    ui::TextureButton2D* new_button = new ui::TextureButton2D(name, "data/textures/material_samples.png", ui::UNIQUE_SELECTION);
    new_button->remove_flag(MATERIAL_2D);
    mat_samples->add_child(new_button);
    num_generated_materials++;

    // Add data to existing samples..
    const StrokeMaterial& mat = stroke_parameters.get_material();
    add_pbr_material_data(name, mat.color, mat.roughness, mat.metallic,
        mat.noise_params.x, mat.noise_color, mat.noise_params.y, static_cast<int>(mat.noise_params.z));

    Node::bind(name, [&](const std::string& signal, void* button) {
        update_stroke_from_material(signal);
    });
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
    glm::mat4x4 pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
    glm::vec3 ray_dir = get_front(pose);
    renderer->get_raymarching_renderer()->octree_ray_intersect(pose[3], ray_dir);

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
