#include "sculpt_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/parsers/parse_gltf.h"
#include "framework/parsers/parse_scene.h"
#include "framework/ui/keyboard.h"
#include "framework/camera/camera.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "graphics/primitives/quad_mesh.h"
#include "graphics/primitives/box_mesh.h"

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/mesh_transparent.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

uint8_t SculptEditor::last_generated_material_uid = 0;

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    const sSDFGlobals& sdf_globals = renderer->get_sdf_globals();
    stroke_manager.set_brick_world_size(glm::vec3(sdf_globals.brick_world_size));

    mirror_mesh = new MeshInstance3D();
    mirror_mesh->set_mesh(new QuadMesh());
    mirror_mesh->scale(glm::vec3(0.25f));

    Material* mirror_material = new Material();
    mirror_material->set_priority(0);
    mirror_material->set_transparency_type(ALPHA_BLEND);
    mirror_material->set_cull_type(CULL_NONE);
    mirror_material->set_diffuse_texture(RendererStorage::get_texture("data/textures/mirror_quad_texture.png"));
    mirror_material->set_type(MATERIAL_UNLIT);
    mirror_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries, mirror_material));

    mirror_mesh->set_surface_material_override(mirror_mesh->get_surface(0), mirror_material);

    axis_lock_gizmo.initialize(TRANSLATE);
    mirror_gizmo.initialize(TRANSLATE);

    // Set maximum number of edits per curve
    current_spline.set_density(MAX_EDITS_PER_SPLINE);

    // Sculpt area box
    {
        sculpt_area_box = parse_mesh("data/meshes/cube.obj");

        Material* sculpt_area_box_material = new Material();
        sculpt_area_box_material->set_priority(0);
        sculpt_area_box_material->set_transparency_type(ALPHA_BLEND);
        sculpt_area_box_material->set_cull_type(CULL_FRONT);
        sculpt_area_box_material->set_type(MATERIAL_UNLIT);
        sculpt_area_box_material->set_diffuse_texture(RendererStorage::get_texture("data/textures/grid_texture.png"));
        sculpt_area_box_material->set_color(colors::RED);
        sculpt_area_box_material->set_shader(RendererStorage::get_shader("data/shaders/sculpt_box_area.wgsl", sculpt_area_box_material));

        sculpt_area_box->set_surface_material_override(sculpt_area_box->get_surface(0), sculpt_area_box_material);

        // Create axis reference

        Surface* s = new Surface();
        s->create_axis(0.08f);

        Material* ref_mat = new Material();
        ref_mat->set_priority(0);
        ref_mat->set_topology_type(TOPOLOGY_LINE_LIST);
        ref_mat->set_transparency_type(ALPHA_BLEND);
        ref_mat->set_type(MATERIAL_UNLIT);
        ref_mat->set_shader(RendererStorage::get_shader("data/shaders/axis.wgsl", ref_mat));

        sculpt_area_box->add_surface(s);
        sculpt_area_box->set_surface_material_override(s, ref_mat);
    }

    // Initialize default primitive states
    {
        primitive_default_states[SD_SPHERE]     = { glm::vec4(0.02f, 0.0f,  0.0f,  0.0f) };
        primitive_default_states[SD_BOX]        = { glm::vec4(0.02f, 0.02f, 0.02f, 0.0f) };
        primitive_default_states[SD_CONE]       = { glm::vec4(0.05f, 0.05f, 0.0f,  0.0f) };
        primitive_default_states[SD_CYLINDER]   = { glm::vec4(0.03f, 0.05f, 0.0f,  0.0f) };
        primitive_default_states[SD_CAPSULE]    = { glm::vec4(0.03f, 0.05f, 0.0f,  0.0f) };
        primitive_default_states[SD_TORUS]      = { glm::vec4(0.03f, 0.01f, 0.0f,  0.0f) };
        primitive_default_states[SD_VESICA]     = { glm::vec4(0.03f, 0.05f, 0.0f,  0.0f) };
    }

    // Edit preview mesh
    {
        Surface* sphere_surface = new Surface();
        sphere_surface->create_sphere();

        mesh_preview = new MeshInstance3D();
        mesh_preview->add_surface(sphere_surface);

        Material* preview_material = new Material();
        preview_material->set_priority(1);
        preview_material->set_transparency_type(ALPHA_BLEND);
        preview_material->set_type(MATERIAL_UNLIT);
        preview_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_transparent::source, shaders::mesh_transparent::path, shaders::mesh_transparent::libraries, preview_material));

        mesh_preview->set_surface_material_override(sphere_surface, preview_material);

        mesh_preview_outline = new MeshInstance3D();
        mesh_preview_outline->add_surface(sphere_surface);

        Material* outline_material = new Material();
        outline_material->set_cull_type(CULL_FRONT);
        outline_material->set_type(MATERIAL_UNLIT);
        outline_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries, outline_material));

        mesh_preview_outline->set_surface_material_override(sphere_surface, outline_material);
    }

    // Default stroke config
    {
        stroke_parameters.set_primitive(SD_SPHERE);
        stroke_parameters.set_operation(OP_SMOOTH_UNION);
        stroke_parameters.set_color_blend_operation(COLOR_OP_REPLACE);
        stroke_parameters.set_parameters({ 0.0f, -1.0f, 1.0f, 0.005f });
        stroke_manager.change_stroke_params(stroke_parameters, 0u);
    }

    // Create UI and bind events
    init_ui();

    // Add pbr materials data
    {
        add_pbr_material_data("aluminium", Color(0.912f, 0.914f, 0.92f, 1.0f), 0.0f, 1.0f);
        add_pbr_material_data("charcoal", Color(0.02f, 0.02f, 0.02f, 1.0f), 0.5f, 0.0f);
        add_pbr_material_data("rusted_iron", Color(0.531f, 0.512f, 0.496f, 1.0f), 0.0f, 1.0f, 1.0f); // add noise
    }

    Node::bind("@on_gpu_results", [&](const std::string& sg, void* data) {

        // Do nothing if it's not the current editor..
        auto engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
        if (engine->get_current_editor() != this) {
            return;
        }

        last_gpu_results = *(reinterpret_cast<sGPU_SculptResults*>(data));

        // Computing sculpt AABB
        {
            const sGPU_SculptResults::sGPU_SculptEvalData& eval_data = last_gpu_results.sculpt_eval_data;

            // If there has been an eval, assign the AABB to the sculpt
            if (static_cast<RoomsRenderer*>(engine->get_renderer())->has_performed_evaluation()) {
                glm::vec3 half_size = (eval_data.aabb_max - eval_data.aabb_min) / 2.0f;
                AABB result = { half_size + eval_data.aabb_min, half_size };
                result = result.transform(Transform::transform_to_mat4(current_sculpt->get_transform()));
                current_sculpt->get_sculpt_data()->set_AABB(result);
            }
        }

        // Computing intersection for surface snap
        {
            const sGPU_RayIntersectionData& intersection = last_gpu_results.ray_intersection;

            if (intersection.has_intersected) {
                last_snap_position = ray_origin + ray_direction * intersection.ray_t;

                if (should_pick_material) {
                    pick_material();
                }
            }
        }
    });

    enable_tool(SCULPT);
}

void SculptEditor::on_enter(void* data)
{
    sculpt_node = reinterpret_cast<SculptNode*>(data);
    assert(sculpt_node);
    set_current_sculpt(sculpt_node);

    RoomsEngine* engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->show_controllers();

    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    Transform& mirror_transform = mirror_gizmo.get_transform();
    Transform& lock_axis_transform = axis_lock_gizmo.get_transform();

    // Get head relative position for setting the sculpt instance if in XR
    if (renderer->get_xr_available()) {
        const AABB sculpt_aabb = sculpt_node->get_sculpt_data()->get_AABB();

        const glm::vec3& to_sculpt_instance_pos = sculpt_node->get_translation();

        has_sculpting_started = stroke_manager.history->size() != 0u;

        // current_instance_transform.set_position(to_sculpt_instance_pos);
        mirror_transform.set_position(to_sculpt_instance_pos);
        lock_axis_transform.set_position(to_sculpt_instance_pos);
    } else {
        mirror_transform.set_position(sculpt_node->get_translation());
        lock_axis_transform.set_position(sculpt_node->get_translation());
    }

    renderer->get_raymarching_renderer()->set_preview_render(true);

    update_ui_workflow_state();
}

void SculptEditor::on_exit()
{
    static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->set_preview_render(false);

    // Remove an empty sculpt on the exit
    if (stroke_manager.history->size() == 0u) {
        current_sculpt->get_sculpt_data()->mark_as_deleted();
    }
}

void SculptEditor::clean()
{
    if(mirror_mesh) delete mirror_mesh;
    if(sculpt_area_box) delete sculpt_area_box;
    if(mesh_preview) delete mesh_preview;
    if(mesh_preview_outline) delete mesh_preview_outline;

    mirror_gizmo.clean();
    axis_lock_gizmo.clean();

    BaseEditor::clean();
}

uint32_t SculptEditor::get_sculpt_context_flags(SculptNode* node)
{
    uint32_t flags = SCULPT_IN_SCULPT_EDITOR;

    if (get_current_sculpt() == node) {
        return flags;
    }

    return (flags | SCULPT_IS_OUT_OF_FOCUS);
}

bool SculptEditor::is_tool_being_used(bool stamp_enabled)
{
#ifdef XR_SUPPORT
    bool is_currently_pressed = !is_something_focused() && Input::is_trigger_pressed(HAND_RIGHT);
    is_released = is_tool_pressed && !is_currently_pressed;

    bool add_edit_with_tool = stamp_enabled ? is_released : is_currently_pressed;

    // Update the is_pressed
    was_tool_pressed = is_tool_pressed;
    is_tool_pressed = is_currently_pressed;

    if (renderer->get_xr_available()) {
        return add_edit_with_tool;
    }
    else {
        return is_picking_material ? Input::was_mouse_pressed(GLFW_MOUSE_BUTTON_LEFT) : Input::was_key_pressed(GLFW_KEY_SPACE);
    }
#else
    return Input::is_key_pressed(GLFW_KEY_SPACE);
#endif
}

bool SculptEditor::edit_update(float delta_time)
{
    stamp_enabled = !is_shift_right_pressed;

    // Poll action using stamp mode when picking material also mode to detect once
    bool is_tool_used = is_tool_being_used(stamp_enabled || is_picking_material);

    if (is_picking_material)
    {
        test_ray_to_sculpts();

        const sGPU_RayIntersectionData& intersection = last_gpu_results.ray_intersection;

        if (intersection.has_intersected) {
            StrokeMaterial tmp = stroke_parameters.get_material();
            last_used_material = tmp;
            tmp.color = Color(intersection.intersection_albedo, 1.0f);
            tmp.roughness = intersection.intersection_roughness;
            tmp.metallic = intersection.intersection_metallic;
            update_gui_from_stroke_material(tmp);
        }

        if (is_tool_used) {
            if (!intersection.has_intersected) {
                update_gui_from_stroke_material(last_used_material);
            }
            should_pick_material = true;
            return false;
        }
    }

    // Move the edit a little away
    glm::mat4x4 controller_pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
    controller_pose = glm::translate(controller_pose, glm::vec3(0.0f, 0.0f, -hand_to_edit_distance));

    // Update edit transform
    if (stroke_mode == STROKE_MODE_STRETCH) {
        edit_to_add.position = edit_position_stamp;
        edit_to_add.rotation = edit_rotation_stamp;
    }
    else {
        edit_to_add.position = controller_pose[3];
        edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT, POSE_AIM));
        edit_position_stamp = edit_to_add.position;
        edit_origin_stamp = edit_to_add.position;
        edit_rotation_stamp = edit_to_add.rotation;
    }

    // Guides: edit position modifiers
    if(renderer->get_xr_available()) {
        // Snap surface

        if (snap_to_surface) {

            test_ray_to_sculpts();

            if (can_snap_to_surface()) {
                edit_to_add.position = last_snap_position;
            }
        }

        if (use_mirror && !hide_mirror) {
            bool r = mirror_gizmo.update(Input::get_controller_position(HAND_RIGHT, POSE_AIM), delta_time);
            is_tool_used &= !r;
            is_tool_pressed &= !r;
            mirror_normal = glm::normalize(mirror_gizmo.get_rotation() * normals::pZ);
        }

        if (snap_to_grid) {
            float grid_multiplier = 1.f / snap_grid_size;
            // Uncomment for grid size of half of the edit radius
            // grid_multiplier = 1.f / (edit_to_add.dimensions.x / 2.f);
            edit_to_add.position = glm::round(edit_to_add.position * grid_multiplier) / grid_multiplier;
        }
        else if (axis_lock) {
            bool r = axis_lock_gizmo.update(Input::get_controller_position(HAND_RIGHT, POSE_AIM), delta_time);
            is_tool_used &= !r;
            is_tool_pressed &= !r;

            glm::vec3 locked_pos = edit_to_add.position;
            if (axis_lock_mode & AXIS_LOCK_X)
                locked_pos.x = axis_lock_gizmo.get_position().x;
            else if (axis_lock_mode & AXIS_LOCK_Y)
                locked_pos.y = axis_lock_gizmo.get_position().y;
            else if (axis_lock_mode & AXIS_LOCK_Z)
                locked_pos.z = axis_lock_gizmo.get_position().z;
            edit_to_add.position = locked_pos;
        }
    }

    // Right Thumbstick logic
    {
        // Disable the unused joystick axis until the joystick is released
        uint8_t curr_thumbstick_axis = Input::get_leading_thumbstick_axis(HAND_RIGHT);

        if (thumbstick_leading_axis == XR_THUMBSTICK_NO_AXIS) {
            thumbstick_leading_axis = curr_thumbstick_axis;
        }
        else if (curr_thumbstick_axis == XR_THUMBSTICK_NO_AXIS) {
            thumbstick_leading_axis = XR_THUMBSTICK_NO_AXIS;
        }

        bool use_x_axis = (thumbstick_leading_axis == XR_THUMBSTICK_AXIS_X);
        const glm::vec2& thumbstick_values = Input::get_thumbstick_value(HAND_RIGHT);
        float size_multiplier = (use_x_axis ? thumbstick_values.x : thumbstick_values.y) * delta_time * 0.1f;

        // Update edit dimensions
        if(stroke_mode == STROKE_MODE_NONE) {

            // Get the data from the primitive default
            edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;

            if (std::abs(size_multiplier) > 0.f) {
                // Update primitive main size
                if (!is_shift_right_pressed) {
                    if (use_x_axis) {
                        // Update rounded size
                        edit_to_add.dimensions.w = glm::clamp(edit_to_add.dimensions.w + size_multiplier, 0.0f, MAX_PRIMITIVE_SIZE);
                    }
                    else {
                        edit_to_add.dimensions.x = glm::clamp(size_multiplier + edit_to_add.dimensions.x, MIN_PRIMITIVE_SIZE, MAX_PRIMITIVE_SIZE);
                        if (stroke_parameters.get_primitive() == SD_BOX) {
                            edit_to_add.dimensions = glm::vec4(glm::vec3(edit_to_add.dimensions.x), edit_to_add.dimensions.w);
                        }
                    }
                }
                else {
                    if (use_x_axis) {
                        // Change smooth factor
                        float current_smooth = stroke_parameters.get_smooth_factor();
                        stroke_parameters.set_smooth_factor(glm::clamp(current_smooth + size_multiplier * 0.2f, MIN_SMOOTH_FACTOR, MAX_SMOOTH_FACTOR));
                        Node::emit_signal("smooth_factor@changed", stroke_parameters.get_smooth_factor());
                    }
                    else {
                        // Update primitive specific size (secondary size)
                        edit_to_add.dimensions.y = glm::clamp(edit_to_add.dimensions.y + size_multiplier, MIN_PRIMITIVE_SIZE, MAX_PRIMITIVE_SIZE);
                    }
                }

                primitive_default_states[stroke_parameters.get_primitive()].dimensions = edit_to_add.dimensions;
                dimensions_dirty = true;
            }
        }
        else if (stroke_mode == STROKE_MODE_SPLINE) {

            // Get the data from the primitive default
            edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;

            if (std::abs(size_multiplier) > 0.f) {

                // Change spline density
                if (use_x_axis) {
                    float current_distance = current_spline.get_knot_distance();
                    // Represented as "density" so the multiplier substracts instead of adding distance
                    current_spline.set_knot_distance(glm::clamp(current_distance - size_multiplier, 0.005f, 0.1f));
                }
                else {
                    // Update primitive uniform size
                    edit_to_add.dimensions = glm::vec4(glm::vec3(glm::clamp(edit_to_add.dimensions + size_multiplier, MIN_PRIMITIVE_SIZE, MAX_PRIMITIVE_SIZE)), edit_to_add.dimensions.w);
                }

                primitive_default_states[stroke_parameters.get_primitive()].dimensions = edit_to_add.dimensions;
                dimensions_dirty = true;
            }
        }
    }

    // Left Thumbstick logic: Change "hand_to_edit_distance" using Y axis
    {
        float lthumbstick_y_value = Input::get_thumbstick_value(HAND_LEFT).y;
        float size_multiplier = lthumbstick_y_value * delta_time * 0.1f;
        hand_to_edit_distance = glm::clamp(hand_to_edit_distance + size_multiplier, 0.01f, 0.1f);
    }

    // Stretch the edit using motion controls
    if (stamp_enabled && is_tool_pressed) {

        if (stroke_mode == STROKE_MODE_STRETCH) {
            sdPrimitive curr_primitive = stroke_parameters.get_primitive();

            const glm::quat& hand_rotation = Input::get_controller_rotation(HAND_RIGHT, POSE_AIM);
            const glm::vec3& hand_position = controller_pose[3];

            const glm::vec3& stamp_origin_to_hand = edit_origin_stamp - hand_position;
            const float stamp_to_hand_distance = glm::length(stamp_origin_to_hand);
            const glm::vec3& stamp_to_hand_norm = stamp_origin_to_hand / (stamp_to_hand_distance);

            // Get rotation of the controller, along the stretch direction
            glm::quat swing, twist;
            quat_swing_twist_decomposition(stamp_to_hand_norm, hand_rotation, swing, twist);
            twist.w *= -1.0f;

            edit_rotation_stamp = get_quat_between_vec3(stamp_origin_to_hand, glm::vec3(0.0f, -stamp_to_hand_distance, 0.0f)) * twist;

            // TODO: Remove when we can support BIG primitives
            float temp_limit = 0.23f;

            switch (curr_primitive)
            {
            case SD_SPHERE:
                edit_to_add.dimensions.x = stamp_to_hand_distance;
                break;
            case SD_BOX:
                edit_position_stamp = edit_origin_stamp - stamp_to_hand_norm * stamp_to_hand_distance * 0.5f;
                edit_to_add.dimensions.y = stamp_to_hand_distance * 0.5f;
                break;
            case SD_CAPSULE:
                edit_to_add.position = edit_origin_stamp;
                edit_to_add.dimensions.y = stamp_to_hand_distance;
                temp_limit *= 2.0f;
                break;
            case SD_CYLINDER:
                edit_position_stamp = edit_origin_stamp - stamp_origin_to_hand * stamp_to_hand_distance * 0.5f;
                edit_to_add.dimensions.y = stamp_to_hand_distance * 0.5f;
                break;
            default:
                break;
            }

            edit_to_add.dimensions = glm::clamp(edit_to_add.dimensions, glm::vec4(MIN_PRIMITIVE_SIZE), glm::vec4(temp_limit));
            dimensions_dirty = true;
        }
#if defined(XR_SUPPORT)
        // Only enter spline mode when the acceleration of the hand exceds a threshold
        // To stretch, toggle with 'B'
        else if(!creating_path && glm::length(glm::abs(controller_movement_data[HAND_RIGHT].velocity)) > 0.30f) {
            start_spline(true);
        }
#endif
    }

    if (!creating_path) {

        stroke_mode = (is_tool_used && !stamp_enabled) ? STROKE_MODE_SMEAR : STROKE_MODE_NONE;

        update_edit_rotation();
    }

    // Debug sculpting
    {
        // For debugging sculpture without a headset
        if (!renderer->get_xr_available()) {

            if (is_tool_being_used(stamp_enabled)) {
                edit_to_add.position = current_sculpt->get_transform().get_position() + glm::vec3(glm::vec3(0.2f * (random_f() * 4 - 2), 0.2f * (random_f() * 4 - 2), 0.2f * (random_f() * 4 - 2)));
                glm::vec3 euler_angles(glm::pi<float>()* random_f(), glm::pi<float>()* random_f(), glm::pi<float>()* random_f());
                // edit_to_add.dimensions = glm::vec4(0.05f, 0.05f, 0.05f, 0.0f) * 1.0f;
                //edit_to_add.dimensions = (edit_to_add.operation == OP_SUBSTRACTION) ? 3.0f * glm::vec4(0.2f, 0.2f, 0.2f, 0.2f) : glm::vec4(0.2f, 0.2f, 0.2f, 0.2f);
                edit_to_add.rotation = glm::normalize(glm::inverse(glm::normalize(glm::quat(euler_angles))));
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
                edit_to_add.position = current_sculpt->get_transform().get_position();
                edit_to_add.rotation = current_sculpt->get_transform().get_rotation();
            }
        }
    }

    // Store now since later it will be converted to 3d texture space
    edit_position_world = edit_to_add.position;
    edit_rotation_world = edit_to_add.rotation;

    // Add edit based on controller movement
    // TODO(Juan): Check rotation?
#if defined(XR_SUPPORT)
    if (!stamp_enabled && was_tool_pressed && is_tool_used) {
        if (glm::length(controller_movement_data[HAND_RIGHT].prev_edit_position - edit_position_world) < (edit_to_add.dimensions.x / 3.0f)) {
            is_tool_used = false;
        } else {
            controller_movement_data[HAND_RIGHT].prev_edit_position = edit_position_world;
        }
    }
    else {
        controller_movement_data[HAND_RIGHT].prev_edit_position = edit_position_world;
    }
#endif

    return is_tool_used;
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    BaseEditor::update(delta_time);

    // Update controller UI
    if (renderer->get_xr_available()) {
        generate_shortcuts();
    }

    preview_tmp_edits.clear();
    new_edits.clear();

    // Operation changer for the different tools
    {
        if (Input::was_button_pressed(XR_BUTTON_B)) {

            if (creating_path) {
                stroke_mode = (stroke_mode == STROKE_MODE_STRETCH) ? STROKE_MODE_SPLINE : STROKE_MODE_STRETCH;
            }
            else if (!is_shift_right_pressed) {
                snap_to_surface = !snap_to_surface;
            }
        }

        if (Input::was_button_pressed(XR_BUTTON_A)) {

            if (creating_path) {
                // do nothing here!
            }
            else if (is_shift_right_pressed) {
                test_ray_to_sculpts();
                should_pick_material = true;
            }
            // Add/Substract toggle
            else {

                sdOperation op = stroke_parameters.get_operation();

                if (current_tool == SCULPT) {
                    switch (op) {
                    case OP_SMOOTH_UNION:
                        op = OP_SMOOTH_SUBSTRACTION;
                        Node::emit_signal("substract@pressed", (void*)nullptr);
                        break;
                    case OP_SMOOTH_SUBSTRACTION:
                        op = OP_SMOOTH_UNION;
                        Node::emit_signal("add@pressed", (void*)nullptr);
                        break;
                    default:
                        assert(0 && "Use smooth operations!");
                        break;
                    }

                    stroke_parameters.set_operation(op);
                }
            }
        }
    }

    // Undo/Redo open ui stuff
    {
        bool x_pressed = Input::was_button_pressed(XR_BUTTON_X);

        if (!is_shift_left_pressed) {
            if (Input::was_key_pressed(GLFW_KEY_U) || x_pressed) {
                undo();
            }
        }
        else if (Input::was_key_pressed(GLFW_KEY_R) || x_pressed) {
            redo();
        }
    }

    if (Input::was_button_pressed(XR_BUTTON_Y) && !creating_path) {
        RoomsEngine::switch_editor(SCENE_EDITOR);
    }

    bool is_tool_used = edit_update(delta_time);
    if (is_tool_used) {
        has_sculpting_started = true;
    }

    if (!has_sculpting_started) {
        sculpt_node->set_position(edit_to_add.position);
    }

    update_sculpt_rotation();

    if (Input::was_button_pressed(XR_BUTTON_B) && !creating_path && is_shift_right_pressed) {
        glm::vec3 texture_offset = world_to_texture3d(current_sculpt->get_translation()) - edit_to_add.position;
        renderer->get_sculpt_manager()->apply_sculpt_offset(current_sculpt, texture_offset);
        stroke_manager.set_current_sculpt(current_sculpt);
    }

    // If any parameter changed or just stopped sculpting change the stroke
    bool must_change_stroke = stroke_parameters.is_dirty();
    must_change_stroke |= (was_tool_pressed && !is_tool_pressed);
    must_change_stroke |= force_new_stroke;

    // Edit & Stroke submission
    {
        if (must_change_stroke) {
            stroke_manager.change_stroke_params(stroke_parameters);
            stroke_parameters.set_dirty(false);
            force_new_stroke = false;
        }

        // Submit spline edits in the next frame..
        if (dirty_spline) {
            current_spline.for_each([&](const Knot& point) {
                Edit edit;
                edit.position = point.position;
                // edit.rotation = point.rotation;
                edit.rotation = current_spline.get_knot(0u).rotation;
                edit.dimensions = glm::vec4(point.size, 0.0f);
                new_edits.push_back(edit);
            });

            add_edit_repetitions(new_edits);

            force_new_stroke = true;
            reset_spline();
        }

        // Manage spline knot additions
        if (creating_spline()) {

            if (!current_spline.size()) {
                current_spline.add_knot({ edit_to_add.position, edit_to_add.dimensions, edit_to_add.rotation });
            }
            else if (Input::was_button_pressed(XR_BUTTON_A)) {
                current_spline.add_knot({ edit_to_add.position, edit_to_add.dimensions, edit_to_add.rotation });

                if (current_spline.size() >= MAX_KNOTS_PER_SPLINE) {
                    end_spline();
                }
            }
        }
        // Upload the edit to the edit list
        else if (is_tool_used) {

            new_edits.push_back(edit_to_add);

            add_edit_repetitions(new_edits);

            // a hack for flatscreen sculpting
            if (!renderer->get_xr_available() && new_edits.size() > 0u) {
                stroke_manager.change_stroke_params(stroke_parameters);
            }

            // Add recent color only when is used...
            add_recent_color(stroke_parameters.get_material().color);
        }
    }

    // Add preview edits

    if (creating_spline()) {

        preview_spline = current_spline;

        preview_spline.add_knot({ edit_to_add.position, edit_to_add.dimensions, edit_to_add.rotation });

        preview_spline.for_each([&](const Knot& point) {
            Edit edit;
            edit.position = point.position;
            edit.rotation = preview_spline.get_knot(0u).rotation;
            // edit.rotation = point.rotation;
            edit.dimensions = glm::vec4(point.size, 0.0f);
            preview_tmp_edits.push_back(edit);
        });
    }
    else {
        preview_tmp_edits.push_back(edit_to_add);
    }

    // Mirror functionality
    if (use_mirror) {
        mirror_current_edits(delta_time);
    }

    set_preview_stroke();

    bool needs_evaluation = false;
    if (called_undo) {
        needs_evaluation = stroke_manager.undo();
        called_undo = false;
    } else if (called_redo) {
        needs_evaluation = stroke_manager.redo();
        called_redo = false;
    } else if (new_edits.size() > 0u) {
        // If there is not history, only add the non-substract edits
        if (stroke_manager.history->size() == 0u) {
            if (stroke_parameters.get_operation() != OP_SMOOTH_SUBSTRACTION) {
                needs_evaluation = stroke_manager.add(new_edits);
            }
        } else {
            needs_evaluation = stroke_manager.add(new_edits);
        }
        
    }

    if (needs_evaluation) {
        renderer->get_sculpt_manager()->update_sculpt(
            current_sculpt->get_sculpt_data(),
            stroke_manager.result_to_compute,
            stroke_manager.edit_list_count,
            stroke_manager.edit_list);
    }

    // Update UI state if:
    // a) evaluated
    // b) tool used (is_released in xr, or is_tool_used in 2d)
    if (needs_evaluation || is_released || (is_tool_used && !renderer->get_xr_available())) {
        update_ui_workflow_state();
    }
    
    was_tool_used = is_tool_used;

    if (is_tool_used) {
        renderer->toogle_frame_debug();

        if (stroke_mode == STROKE_MODE_STRETCH) {
            edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;
            dimensions_dirty = true;
            reset_spline();
        } else if (creating_spline()) {
            // Add last position
            current_spline.add_knot({ edit_to_add.position, edit_to_add.dimensions, edit_to_add.rotation });
            end_spline();
        }

        stroke_mode = STROKE_MODE_NONE;
    }
}

void SculptEditor::set_preview_stroke()
{
    // Add repetitions before setting preview stroke
    add_edit_repetitions(preview_tmp_edits);

    sGPUStroke preview_stroke;

    preview_stroke.color_blending_op = stroke_parameters.get_color_blend_operation();
    preview_stroke.primitive = stroke_parameters.get_primitive();
    preview_stroke.material = stroke_parameters.get_material();
    preview_stroke.operation = stroke_parameters.get_operation();
    preview_stroke.parameters = stroke_parameters.get_parameters();
    preview_stroke.edit_count = preview_tmp_edits.size();

    AABB stroke_aabb = preview_stroke.get_world_AABB_of_edit_list(preview_tmp_edits);
    preview_stroke.aabb_min = stroke_aabb.center - stroke_aabb.half_size;
    preview_stroke.aabb_max = stroke_aabb.center + stroke_aabb.half_size;

    renderer->get_sculpt_manager()->set_preview_stroke(
        current_sculpt->get_sculpt_data(),
        current_sculpt->get_in_frame_model_idx(),
        preview_stroke, preview_tmp_edits
    );
}

void SculptEditor::add_edit_repetitions(std::vector<Edit>& edits)
{
    if (rep_count == 0u) {
        return;
    }

    size_t edit_count = edits.size();

    float og_offset = rep_count * rep_spacing * 0.5f;

    for (size_t i = 0u; i < edit_count; i++) {

        Edit& edit = edits[i];
        edit.position -= (glm::vec3(og_offset, 0.0f, 0.0f) * edit.rotation);

        for (uint8_t k = 1u; k <= rep_count; k++) {
            Edit rep_edit = edit;
            rep_edit.position += (glm::vec3(rep_spacing * k, 0.0f, 0.0f) * edit.rotation);
            edits.push_back(rep_edit);
        }
    }
}

void SculptEditor::apply_mirror_position(glm::vec3& position)
{
    // Don't rotate the mirror origin..
    glm::vec3 origin_texture_space = world_to_texture3d(mirror_gizmo.get_position());
    glm::vec3 normal_texture_space = world_to_texture3d(mirror_normal, true);
    glm::vec3 pos_to_origin = origin_texture_space - position;
    glm::vec3 reflection = glm::reflect(pos_to_origin, normal_texture_space);
    position = origin_texture_space - reflection;
}

void SculptEditor::apply_mirror_rotation(glm::quat& rotation) const
{
    glm::vec3 curr_dir = rotation * normals::pZ;
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
        pos_texture_space -= (current_sculpt->get_transform().get_position());
    }

    pos_texture_space = glm::inverse(current_sculpt->get_transform().get_rotation()) * pos_texture_space;

    return pos_texture_space;
}

glm::vec3 SculptEditor::texture3d_to_world(const glm::vec3& position)
{
    glm::vec3 pos_world_space;

    pos_world_space = glm::inverse(current_sculpt->get_transform().get_rotation()) * position;
    pos_world_space = pos_world_space + (current_sculpt->get_transform().get_position());

    return pos_world_space;
}

void SculptEditor::test_ray_to_sculpts()
{
    // Send rays each frame to detect hovered sculpts and other nodes
    Engine::instance->get_scene_ray(ray_origin, ray_direction);

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    rooms_renderer->get_sculpt_manager()->set_ray_to_test(ray_origin, ray_direction, current_sculpt->get_sculpt_data(), current_sculpt->get_in_frame_model_idx());
}

void SculptEditor::update_sculpt_rotation()
{
    if (Input::was_key_pressed(GLFW_KEY_R)) {
        current_sculpt->get_transform().translate(glm::vec3(0.5, 0.0f, 0.0f));
    }

    // Do not rotate sculpt if shift -> we might be rotating the edit
    if (is_rotation_being_used() && !is_shift_left_pressed) {

        glm::quat current_hand_rotation = (Input::get_controller_rotation(HAND_LEFT));
        glm::vec3 current_hand_translation = Input::get_controller_position(HAND_LEFT);

        if (!rotation_started) {
            last_hand_rotation = current_hand_rotation;
            last_hand_translation = current_hand_translation;
        }

        glm::quat rotation_diff = (current_hand_rotation * glm::inverse(last_hand_rotation));
        glm::vec3 translation_diff = current_hand_translation - last_hand_translation;

        const glm::vec3 sculpt_pos_without_rot = current_sculpt->get_transform().get_position() + translation_diff;
        current_sculpt->get_transform().set_position(rotation_diff * (sculpt_pos_without_rot - current_hand_translation) + current_hand_translation);
        current_sculpt->get_transform().rotate_world(rotation_diff);

        rotation_started = true;

        last_hand_rotation = current_hand_rotation;
        last_hand_translation = current_hand_translation;
    }
    else {
        // If rotation has stopped
        if (rotation_started && !is_shift_left_pressed) {
            rotation_started = false;
        }
    }

    // Push edits in 3d texture space
    edit_to_add.position = world_to_texture3d(edit_to_add.position);
    edit_to_add.rotation *= (current_sculpt->get_transform().get_rotation());
}

void SculptEditor::update_edit_rotation()
{
    // Rotate edit only if pressing shift
    if (is_rotation_being_used() && is_shift_left_pressed) {

        glm::quat current_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
        glm::vec3 current_hand_translation = Input::get_controller_position(HAND_LEFT);

        if (!edit_rotation_started) {
            last_hand_rotation = current_hand_rotation;
            last_hand_translation = current_hand_translation;
        }

        edit_rotation_diff = current_hand_rotation * glm::inverse(last_hand_rotation);
        edit_rotation_started = true;
    }

    // If rotation has stopped
    else if (edit_rotation_started) {
        edit_user_rotation = edit_user_rotation * edit_rotation_diff;
        edit_rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
        edit_rotation_started = false;

        sdPrimitive primitive = stroke_parameters.get_primitive();

        primitive_default_states[primitive].rotation = edit_user_rotation;
    }

    glm::quat tmp_rotation = glm::inverse(edit_user_rotation * edit_rotation_diff);
    edit_to_add.rotation = glm::conjugate(tmp_rotation) * edit_to_add.rotation;
}

void SculptEditor::update_ui_workflow_state()
{
    // Undo/Redo buttons
    {
        bool can_undo = stroke_manager.can_undo();

        if (creating_spline()) {
            can_undo = (current_spline.size() > 0u);
        }

        auto b_undo = static_cast<ui::Button2D*>(Node2D::get_widget_from_name("undo"));
        b_undo->set_disabled(!can_undo);

        // TODO: Spline knot cannot be redone by now!
        bool can_redo = stroke_manager.can_redo() && !creating_spline();
        auto b_redo = static_cast<ui::Button2D*>(Node2D::get_widget_from_name("redo"));
        b_redo->set_disabled(!can_redo);
    }
}

void SculptEditor::undo()
{
    if (creating_spline()) {
        current_spline.pop_knot();
        return;
    }
    else {
        renderer->toogle_frame_debug();
        called_undo = true;
    }
}

void SculptEditor::redo()
{
    if (creating_spline()) {
        return;
    }

    renderer->toogle_frame_debug();
    called_redo = true;
}

void SculptEditor::render()
{
    if (mesh_preview) {

        update_edit_preview(edit_to_add.dimensions);

        // Render mesh preview only in XR
        if (renderer->get_xr_available()) {

            // Render something to be able to cull faces later...
            if (!must_render_mesh_preview_outline()) {
                mesh_preview->render();
            }
            else {
                mesh_preview_outline->set_transform(mesh_preview->get_transform());
                mesh_preview_outline->render();
            }
        }
    }

    if (axis_lock) {
        axis_lock_gizmo.render();

        mirror_mesh->set_transform(Transform::identity());
        mirror_mesh->translate(axis_lock_gizmo.get_position());

        if (axis_lock_mode & AXIS_LOCK_X)
            mirror_mesh->rotate(glm::radians(90.0f), normals::pY);
        else if (axis_lock_mode & AXIS_LOCK_Y)
            mirror_mesh->rotate(glm::radians(90.0f), normals::pX);

        mirror_mesh->render();
    }
    else if (use_mirror && !hide_mirror) {

        mirror_gizmo.render();

        if (!renderer->get_xr_available()) {
            mirror_normal = glm::normalize(mirror_gizmo.get_rotation() * normals::pZ);
        }

        mirror_mesh->set_transform(mirror_gizmo.get_transform());
        mirror_mesh->scale(glm::vec3(0.5f));
        mirror_mesh->render();
    }

    BaseEditor::render();

    // Render always or only XR?
    sculpt_area_box->set_transform(Transform::identity());
    sculpt_area_box->translate(current_sculpt->get_transform().get_position());
    sculpt_area_box->scale(glm::vec3(SCULPT_MAX_SIZE * 0.5f));
    sculpt_area_box->rotate(current_sculpt->get_transform().get_rotation());
    sculpt_area_box->render();
}

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

    if (changed) {
        stroke_parameters.set_dirty(true);
    }

    //ImGui::Text("Sculpt Instance Transform");

    //glm::vec3 position = current_instance_transform.get_position();
    //glm::quat rotation = current_instance_transform.get_rotation();
    //glm::vec3 scale = current_instance_transform.get_scale();

    //if (ImGui::DragFloat3("Translation", &position[0], 0.1f)) {
    //    current_instance_transform.set_position(position);
    //}

    //if (ImGui::DragFloat4("Rotation", &rotation[0], 0.1f)) {
    //    current_instance_transform.set_rotation(rotation);
    //}

    //if (ImGui::DragFloat3("Scale", &scale[0], 0.1f)) {
    //    current_instance_transform.set_scale(scale);
    //}
}

bool SculptEditor::can_snap_to_surface()
{
    return snap_to_surface && last_gpu_results.ray_intersection.has_intersected && (stamp_enabled || current_tool == PAINT);
}

bool SculptEditor::must_render_mesh_preview_outline()
{
    return stroke_parameters.get_operation() == OP_UNION || stroke_parameters.get_operation() == OP_SMOOTH_UNION;
}

void SculptEditor::update_edit_preview(const glm::vec4& dims)
{
    // Recreate mesh depending on primitive parameters
    if (dimensions_dirty)
    {
        glm::vec4 new_dims = dims + 0.002f;

        switch (stroke_parameters.get_primitive())
        {
        case SD_SPHERE:
            mesh_preview->get_surface(0)->create_sphere(new_dims.x);
            break;
        case SD_BOX:
            mesh_preview->get_surface(0)->create_rounded_box(new_dims.x, new_dims.y, new_dims.z, glm::clamp(dims.w / 0.08f, 0.0f, 1.0f) * glm::min(new_dims.x, glm::min(new_dims.y, new_dims.z)));
            break;
        case SD_CAPSULE:
            mesh_preview->get_surface(0)->create_capsule(new_dims.x, new_dims.y);
            break;
        case SD_CONE:
            mesh_preview->get_surface(0)->create_cone(new_dims.x, new_dims.y);
            break;
        case SD_CYLINDER:
            mesh_preview->get_surface(0)->create_cylinder(new_dims.x, new_dims.y * 2.f);
            break;
        case SD_TORUS:
            mesh_preview->get_surface(0)->create_torus(new_dims.x, glm::clamp(new_dims.y, 0.0001f, dims.x));
            break;
        default:
            break;
        }

        spdlog::trace("Edit mesh preview generated!");

        Node::emit_signal("main_size@changed", edit_to_add.dimensions.x);
        Node::emit_signal("secondary_size@changed", edit_to_add.dimensions.y);
        Node::emit_signal("round_size@changed", edit_to_add.dimensions.w);

        dimensions_dirty = false;
    }

    glm::mat4x4 preview_pose = glm::translate(glm::mat4x4(1.0f), edit_position_world);
    preview_pose *= glm::inverse(glm::toMat4(edit_rotation_world));

    // Update model depending on the primitive
    switch (stroke_parameters.get_primitive())
    {
    case SD_CAPSULE:
        preview_pose = glm::translate(preview_pose, { 0.f, dims.y * 0.5, 0.f });
        break;
    default:
        break;
    }

    // Update edit transform
    mesh_preview->set_transform(Transform::mat4_to_transform(preview_pose));
}

void SculptEditor::set_primitive(sdPrimitive primitive)
{
    stroke_parameters.set_primitive(primitive);
    dimensions_dirty = true;

    if (primitive_default_states.contains(primitive)) {
        edit_to_add.dimensions = primitive_default_states[primitive].dimensions;
        edit_user_rotation = primitive_default_states[primitive].rotation;
    }
}

void SculptEditor::set_operation(sdOperation operation)
{
    stroke_parameters.set_operation(operation);
}

void SculptEditor::set_edit_size(float main, float secondary, float round)
{
    // Update primitive main size
    if (main >= 0.0f) {
        edit_to_add.dimensions.x = main;
    }
    // Update primitive specific size
    else if (secondary >= 0.0f) {
        edit_to_add.dimensions.y = secondary;
    }
    // Update round size
    else if (round >= 0.0f) {
        edit_to_add.dimensions.w = round;
    }

    // Update in primitive state
    primitive_default_states[stroke_parameters.get_primitive()].dimensions = edit_to_add.dimensions;

    dimensions_dirty = true;
}

void SculptEditor::set_onion_modifier(float value)
{
    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.x = glm::clamp(value, 0.0f, 1.0f);
    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::set_cap_modifier(float value)
{
    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.y = glm::clamp(value, 0.0f, 1.0f);
    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::set_current_sculpt(SculptNode* sculpt_instance)
{
    current_sculpt = sculpt_instance;

    if (current_sculpt) {
        stroke_manager.new_history_add(&current_sculpt->get_sculpt_data()->get_stroke_history());
    }
}

void SculptEditor::enable_tool(eTool tool)
{
    current_tool = tool;

    Node2D* brush_editor_submenu = Node2D::get_widget_from_name("brush_editor");
    assert(brush_editor_submenu);

    switch (tool)
    {
    case SCULPT:
        stroke_parameters.set_operation(OP_SMOOTH_UNION);
        static_cast<ui::ButtonSubmenu2D*>(brush_editor_submenu)->set_disabled(true);
        Node::emit_signal("add@pressed", (void*)nullptr);
        break;
    case PAINT:
        stroke_parameters.set_operation(OP_SMOOTH_PAINT);
        static_cast<ui::ButtonSubmenu2D*>(brush_editor_submenu)->set_disabled(false);
        Node::emit_signal("paint@pressed", (void*)nullptr);
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

/*
*   Splines stuff
*/

void SculptEditor::start_spline(bool update_ui)
{
    if (creating_spline()) {
        reset_spline(false);
    }
    else {
        stroke_mode = STROKE_MODE_SPLINE;
        current_spline.clear();
        creating_path = true;
    }
}

void SculptEditor::reset_spline(bool update_ui)
{
    dirty_spline = false;
    creating_path = false;
    stroke_mode = STROKE_MODE_NONE;
    current_spline.clear();
}

void SculptEditor::end_spline()
{
    dirty_spline = true;

    update_ui_workflow_state();
}

void SculptEditor::toggle_mirror()
{
    use_mirror = !use_mirror;

    // Center gizmo in sculpt on enable
    if (use_mirror) {
        mirror_gizmo.set_transform(current_sculpt->get_transform());
    }
}

/*
*   UI stuff
*/

void SculptEditor::generate_shortcuts()
{
    std::unordered_map<uint8_t, bool> shortcuts;

    shortcuts[shortcuts::MANIPULATE_SCULPT] = true;
    shortcuts[shortcuts::MODIFY_SMOOTH] = is_shift_right_pressed;
    shortcuts[shortcuts::REDO] = is_shift_left_pressed;
    shortcuts[shortcuts::UNDO] = !is_shift_left_pressed;

    if (creating_spline()) {
        shortcuts[shortcuts::ADD_KNOT] = true;
        shortcuts[shortcuts::TOGGLE_STRETCH_SPLINE] = true;
        shortcuts[shortcuts::SPLINE_DENSITY] = !is_shift_right_pressed;
    }
    else {
        shortcuts[shortcuts::CENTER_SCULPT] = is_shift_right_pressed;
        shortcuts[shortcuts::BACK_TO_SCENE] = true;
        shortcuts[shortcuts::MAIN_SIZE] = !is_shift_right_pressed;
        shortcuts[shortcuts::SECONDARY_SIZE] = is_shift_right_pressed;
        shortcuts[shortcuts::ADD_SUBSTRACT] = !is_shift_right_pressed;
        shortcuts[shortcuts::ROUND_SHAPE] = !is_shift_right_pressed;
        shortcuts[shortcuts::SNAP_SURFACE] = !is_shift_right_pressed;
        shortcuts[shortcuts::PICK_MATERIAL] = is_shift_right_pressed;
        shortcuts[shortcuts::STAMP] = !is_shift_right_pressed;
        shortcuts[shortcuts::SMEAR] = is_shift_right_pressed;
    }

    BaseEditor::update_shortcuts(shortcuts);
}

void SculptEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel = new ui::HContainer2D("sculpt_editor_root", { 48.0f, screen_size.y - 224.f }, ui::CREATE_3D);

    Node::bind("sculpt_editor_root@resize", (FuncUVec2)[&](const std::string& signal, glm::u32vec2 window_size) {
        main_panel->set_position({ 48.0f, window_size.y - 224.f });
    });

    const StrokeMaterial& stroke_material = stroke_parameters.get_material();

    // Color picker...

    {
        ui::ColorPicker2D* color_picker = new ui::ColorPicker2D("color_picker", stroke_material.color);

        // Add recent colors around the picker
        {
            size_t child_count = 5;
            float sample_size = 32.0f;
            float full_size = color_picker->get_size().x + sample_size;
            float radius = full_size * 0.515f;

            glm::vec2 center = glm::vec2((-color_picker->get_size().x + sample_size) * 0.5f);

            for (size_t i = 0; i < child_count; ++i)
            {
                float angle = PI - PI_2 * i / (float)(child_count - 1);
                glm::vec2 translation = glm::vec2(radius * cos(angle), radius * sin(angle)) - center;
                ui::Button2D* child = new ui::Button2D("recent_color_" + std::to_string(i), { .position = translation, .size = glm::vec2(sample_size), .color = colors::WHITE });
                child->set_position(translation);
                color_picker->add_child(child);
            }
        }

        main_panel->add_child(color_picker);
    }

    ui::VContainer2D* vertical_container = new ui::VContainer2D("vertical_container", { 0.0f, 0.0f });
    main_panel->add_child(vertical_container);

    // And two main rows
    ui::HContainer2D* first_row = new ui::HContainer2D("main_first_row", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    ui::HContainer2D* second_row = new ui::HContainer2D("main_second_row", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    // ** Primitives **
    {
        ui::ButtonSelector2D* prim_selector = new ui::ButtonSelector2D("shapes", { "data/textures/primitives.png" });
        prim_selector->add_child(new ui::TextureButton2D("sphere", { "data/textures/sphere.png", ui::SELECTED }));
        prim_selector->add_child(new ui::TextureButton2D("cube", { "data/textures/cube.png" }));
        prim_selector->add_child(new ui::TextureButton2D("cone", { "data/textures/cone.png" }));
        prim_selector->add_child(new ui::TextureButton2D("capsule", { "data/textures/capsule.png" }));
        prim_selector->add_child(new ui::TextureButton2D("cylinder", { "data/textures/cylinder.png" }));
        prim_selector->add_child(new ui::TextureButton2D("torus", { "data/textures/torus.png" }));
        prim_selector->add_child(new ui::TextureButton2D("vesica", { "data/textures/x.png" }));
        first_row->add_child(prim_selector);
    }

    {
        ui::ItemGroup2D* g_reps = new ui::ItemGroup2D("g_reps");
        g_reps->add_child(new ui::IntSlider2D("rep_count", { .path = "data/textures/edit_repetition.png", .flags = ui::USER_RANGE, .ivalue_max = 8, .p_data = &rep_count }));
        // in meters
        g_reps->add_child(new ui::FloatSlider2D("rep_spacing", { .path = "data/textures/edit_repetition_spacing.png", .flags = ui::USER_RANGE, .fvalue_max = 0.1f, .precision = 2, .p_data = &rep_spacing }));
        first_row->add_child(g_reps);
    }

    // ** Shape, Brush, Material Editors **
    {
        ui::ItemGroup2D* g_editors = new ui::ItemGroup2D("g_editors");

        // Shape editor
        {
            ui::ButtonSubmenu2D* shape_editor_submenu = new ui::ButtonSubmenu2D("shape_editor", { "data/textures/shape_editor.png" });

            // Edit sizes
            {
                ui::ItemGroup2D* g_edit_sizes = new ui::ItemGroup2D("g_edit_sizes");
                g_edit_sizes->add_child(new ui::FloatSlider2D("main_size", { .fvalue = edit_to_add.dimensions.x, .mode = ui::SliderMode::HORIZONTAL, .fvalue_min = MIN_PRIMITIVE_SIZE, .fvalue_max = MAX_PRIMITIVE_SIZE, .precision = 3 }));
                g_edit_sizes->add_child(new ui::FloatSlider2D("secondary_size", { .fvalue = edit_to_add.dimensions.y, .mode = ui::SliderMode::HORIZONTAL, .fvalue_min = MIN_PRIMITIVE_SIZE, .fvalue_max = MAX_PRIMITIVE_SIZE, .precision = 3 }));
                g_edit_sizes->add_child(new ui::FloatSlider2D("round_size", { .path = "data/textures/rounding.png", .fvalue = edit_to_add.dimensions.w, .fvalue_max = MAX_PRIMITIVE_SIZE, .precision = 2 }));
                shape_editor_submenu->add_child(g_edit_sizes);
            }

            // Edit modifiers
            {
                ui::ItemGroup2D* g_edit_modifiers = new ui::ItemGroup2D("g_edit_modifiers");
                //g_edit_modifiers->add_child(new ui::FloatSlider2D("onion_value", { .path = "data/textures/onion.png", .fvalue = stroke_parameters.get_parameters().x }));
                g_edit_modifiers->add_child(new ui::FloatSlider2D("cap_value", { .path = "data/textures/capped.png", .fvalue = stroke_parameters.get_parameters().y }));
                shape_editor_submenu->add_child(g_edit_modifiers);
            }

            g_editors->add_child(shape_editor_submenu);
        }

        // Material editor
        {
            ui::ButtonSubmenu2D* material_editor_submenu = new ui::ButtonSubmenu2D("material_editor", { "data/textures/material_editor.png" });

            // Shading properties
            /*{
                ui::ButtonSubmenu2D* shading_submenu = new ui::ButtonSubmenu2D("shading", "data/textures/shading.png");

                {
                    ui::ItemGroup2D* g_edit_pbr = new ui::ItemGroup2D("g_edit_pbr");
                    g_edit_pbr->add_child(new ui::FloatSlider2D("roughness", 0.7f));
                    g_edit_pbr->add_child(new ui::FloatSlider2D("metallic", 0.2f));
                    shading_submenu->add_child(g_edit_pbr);
                }

                {
                    ui::ItemGroup2D* g_edit_pattern = new ui::ItemGroup2D("g_edit_pattern");
                    g_edit_pattern->add_child(new ui::FloatSlider2D("noise_intensity", 0.0f, ui::SliderMode::VERTICAL, 0.0f, 10.0f));
                    g_edit_pattern->add_child(new ui::FloatSlider2D("noise_frequency", 20.0f, ui::SliderMode::VERTICAL, 0.0f, 50.0f));
                    g_edit_pattern->add_child(new ui::FloatSlider2D("noise_octaves", 8.0f, ui::SliderMode::VERTICAL, 0.0f, 16.0f, 1.0f));
                    g_edit_pattern->add_child(new ui::ColorPicker2D("noise_color_picker", colors::WHITE));
                    material_editor_submenu->add_child(g_edit_pattern);
                }

                material_editor_submenu->add_child(shading_submenu);
            }*/

            // Put directly these two props until there are more pbr props to show
            ui::ItemGroup2D* g_edit_pbr = new ui::ItemGroup2D("g_edit_pbr");
            g_edit_pbr->add_child(new ui::FloatSlider2D("roughness", { .path = "data/textures/roughness.png", .fvalue = stroke_material.roughness }));
            g_edit_pbr->add_child(new ui::FloatSlider2D("metallic", { .path = "data/textures/metallic.png", .fvalue = stroke_material.metallic }));
            material_editor_submenu->add_child(g_edit_pbr);

            // Shuffle
            {
                material_editor_submenu->add_child(new ui::TextureButton2D("shuffle_material", { "data/textures/shuffle.png" }));
            }

            // Materials: add, pick, defaults
            {
                ui::ItemGroup2D* g_saved_materials = new ui::ItemGroup2D("g_saved_materials");

                g_saved_materials->add_child(new ui::TextureButton2D("save_material", { "data/textures/add.png" }));
                g_saved_materials->add_child(new ui::TextureButton2D("pick_material", { "data/textures/pick_material.png", ui::ALLOW_TOGGLE }));

                {
                    ui::ButtonSelector2D* mat_samples_selector = new ui::ButtonSelector2D("material_samples", { "data/textures/material_samples.png" });
                    mat_samples_selector->add_child(new ui::TextureButton2D("aluminium", { "data/textures/material_samples.png", ui::UNIQUE_SELECTION }));
                    mat_samples_selector->add_child(new ui::TextureButton2D("charcoal", { "data/textures/material_samples.png", ui::UNIQUE_SELECTION }));
                    mat_samples_selector->add_child(new ui::TextureButton2D("rusted_iron", { "data/textures/material_samples.png", ui::UNIQUE_SELECTION }));
                    g_saved_materials->add_child(mat_samples_selector);
                }

                material_editor_submenu->add_child(g_saved_materials);
            }

            g_editors->add_child(material_editor_submenu);
        }

        // Brush editor
        {

            ui::ButtonSubmenu2D* brush_editor_submenu = new ui::ButtonSubmenu2D("brush_editor", { "data/textures/brush_editor.png", ui::DISABLED });

            brush_editor_submenu->add_child(new ui::FloatSlider2D("color_override", { .path = "data/textures/r.png", .fvalue = stroke_parameters.get_parameters().z }));

            {
                ui::ButtonSelector2D* color_blend_selector = new ui::ButtonSelector2D("color_blend", { "data/textures/color_blend.png" });
                color_blend_selector->add_child(new ui::TextureButton2D("replace", { "data/textures/r.png" }));
                color_blend_selector->add_child(new ui::TextureButton2D("mix", { "data/textures/x.png" }));
                color_blend_selector->add_child(new ui::TextureButton2D("additive", { "data/textures/a.png" }));
                color_blend_selector->add_child(new ui::TextureButton2D("multiply", { "data/textures/m.png" }));
                color_blend_selector->add_child(new ui::TextureButton2D("screen", { "data/textures/s.png" }));
                brush_editor_submenu->add_child(color_blend_selector);
            }

            g_editors->add_child(brush_editor_submenu);
        }

        first_row->add_child(g_editors);
    }

    // ** Guides **
    {
        ui::ButtonSubmenu2D* guides_submenu = new ui::ButtonSubmenu2D("guides", { "data/textures/mirror.png" });
        ui::ItemGroup2D* g_guides = new ui::ItemGroup2D("g_guides");

        // Mirror
        {
            ui::ButtonSubmenu2D* mirror_submenu = new ui::ButtonSubmenu2D("mirror", { "data/textures/mirror.png" });
            mirror_submenu->add_child(new ui::TextureButton2D("use_mirror", { "data/textures/mirror.png", ui::ALLOW_TOGGLE }));
            mirror_submenu->add_child(new ui::TextureButton2D("hide_mirror", { "data/textures/cross.png", ui::ALLOW_TOGGLE }));
            ui::ComboButtons2D* g_mirror = new ui::ComboButtons2D("g_mirror");
            g_mirror->add_child(new ui::TextureButton2D("mirror_translation", { "data/textures/translation_gizmo.png", ui::SELECTED }));
            g_mirror->add_child(new ui::TextureButton2D("mirror_rotation", { "data/textures/rotation_gizmo.png" }));
            g_mirror->add_child(new ui::TextureButton2D("mirror_both", { "data/textures/transform_gizmo.png" }));
            mirror_submenu->add_child(g_mirror);
            g_guides->add_child(mirror_submenu);
        }

        // Snap to surface, grid
        g_guides->add_child(new ui::TextureButton2D("snap_to_surface", { "data/textures/snap_to_surface.png", ui::ALLOW_TOGGLE }));
        g_guides->add_child(new ui::TextureButton2D("snap_to_grid", { "data/textures/snap_to_grid.png", ui::ALLOW_TOGGLE }));

        // Snap to axis
        {
            ui::ButtonSubmenu2D* lock_axis_submenu = new ui::ButtonSubmenu2D("lock_axis", { "data/textures/lock_axis.png" });
            lock_axis_submenu->add_child(new ui::TextureButton2D("lock_axis_toggle", { "data/textures/lock_axis.png", ui::ALLOW_TOGGLE }));
            ui::ComboButtons2D* g_lock_axis = new ui::ComboButtons2D("g_lock_axis");
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_x", { "data/textures/x.png" }));
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_y", { "data/textures/y.png" }));
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_z", { "data/textures/z.png", ui::SELECTED }));
            lock_axis_submenu->add_child(g_lock_axis);
            g_guides->add_child(lock_axis_submenu);
        }

        guides_submenu->add_child(g_guides);
        first_row->add_child(guides_submenu);
    }

    // ** Go back to scene editor **
    second_row->add_child(new ui::TextureButton2D("go_back", { "data/textures/back.png" }));

    // ** Main tools (SCULPT & PAINT) **
    {
        ui::ComboButtons2D* combo_main_tools = new ui::ComboButtons2D("combo_main_tools");
        combo_main_tools->add_child(new ui::TextureButton2D("add", { "data/textures/cube_add.png", ui::SELECTED }));
        combo_main_tools->add_child(new ui::TextureButton2D("substract", { "data/textures/cube_substract.png" }));
        combo_main_tools->add_child(new ui::TextureButton2D("paint", { "data/textures/paint.png" }));
        second_row->add_child(combo_main_tools);
    }

    // Smooth factor
    {
        ui::Slider2D* smooth_factor_slider = new ui::FloatSlider2D("smooth_factor", { .path = "data/textures/smooth.png", .fvalue = stroke_parameters.get_smooth_factor(), .flags = ui::SKIP_VALUE,
            .fvalue_min = MIN_SMOOTH_FACTOR, .fvalue_max = MAX_SMOOTH_FACTOR, .precision = 3 });
        second_row->add_child(smooth_factor_slider);
    }

    // ** Undo/Redo **
    {
        second_row->add_child(new ui::TextureButton2D("undo", { "data/textures/undo.png" }));
        second_row->add_child(new ui::TextureButton2D("redo", { "data/textures/redo.png" }));
    }

    // Load controller UI labels
    if (renderer->get_xr_available())
    {
        // Thumbsticks
        // Buttons
        // Triggers

        glm::vec2 double_size = { 2.0f, 1.0f };

        // Left hand
        {
            left_hand_box = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            left_hand_box->add_child(new ui::ImageLabel2D("Back to scene", shortcuts::Y_BUTTON_PATH, shortcuts::BACK_TO_SCENE));
            left_hand_box->add_child(new ui::ImageLabel2D("Redo", shortcuts::L_GRIP_X_BUTTON_PATH, shortcuts::REDO, double_size));
            left_hand_box->add_child(new ui::ImageLabel2D("Undo", shortcuts::X_BUTTON_PATH, shortcuts::UNDO));
            left_hand_box->add_child(new ui::ImageLabel2D("Move Sculpt", shortcuts::L_TRIGGER_PATH, shortcuts::MANIPULATE_SCULPT));
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            right_hand_box->add_child(new ui::ImageLabel2D("Main size", shortcuts::R_THUMBSTICK_Y_PATH, shortcuts::MAIN_SIZE));
            right_hand_box->add_child(new ui::ImageLabel2D("Sec size", shortcuts::R_GRIP_R_THUMBSTICK_Y_PATH, shortcuts::SECONDARY_SIZE, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Round Shape", shortcuts::R_THUMBSTICK_X_PATH, shortcuts::ROUND_SHAPE));
            right_hand_box->add_child(new ui::ImageLabel2D("Smooth", shortcuts::R_GRIP_R_THUMBSTICK_X_PATH, shortcuts::MODIFY_SMOOTH, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Spline Density", shortcuts::R_THUMBSTICK_X_PATH, shortcuts::SPLINE_DENSITY));
            right_hand_box->add_child(new ui::ImageLabel2D("Stretch/Spline", shortcuts::B_BUTTON_PATH, shortcuts::TOGGLE_STRETCH_SPLINE));
            right_hand_box->add_child(new ui::ImageLabel2D("Surface Snap", shortcuts::B_BUTTON_PATH, shortcuts::SNAP_SURFACE));
            right_hand_box->add_child(new ui::ImageLabel2D("Center Sculpt", shortcuts::R_GRIP_B_BUTTON_PATH, shortcuts::CENTER_SCULPT, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Add/Substract", shortcuts::A_BUTTON_PATH, shortcuts::ADD_SUBSTRACT));
            right_hand_box->add_child(new ui::ImageLabel2D("Add Knot", shortcuts::A_BUTTON_PATH, shortcuts::ADD_KNOT));
            right_hand_box->add_child(new ui::ImageLabel2D("Pick Material", shortcuts::R_GRIP_A_BUTTON_PATH, shortcuts::PICK_MATERIAL, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Stamp", shortcuts::R_TRIGGER_PATH, shortcuts::STAMP));
            right_hand_box->add_child(new ui::ImageLabel2D("Smear", shortcuts::R_GRIP_R_TRIGGER_PATH, shortcuts::SMEAR, double_size));
        }
    }

    // Bind callbacks
    bind_events();
}

void SculptEditor::bind_events()
{
    Node::bind("go_back", [&](const std::string& signal, void* button) {
        RoomsEngine::switch_editor(SCENE_EDITOR);
    });

    Node::bind("add", [&](const std::string& signal, void* button) {
        enable_tool(SCULPT);
        set_operation(OP_SMOOTH_UNION);
    });

    Node::bind("substract", [&](const std::string& signal, void* button) {
        enable_tool(SCULPT);
        set_operation(OP_SMOOTH_SUBSTRACTION);
    });

    Node::bind("paint", [&](const std::string& signal, void* button) { enable_tool(PAINT); });

    Node::bind("sphere", [&](const std::string& signal, void* button) {  set_primitive(SD_SPHERE); });
    Node::bind("cube", [&](const std::string& signal, void* button) { set_primitive(SD_BOX); });
    Node::bind("cone", [&](const std::string& signal, void* button) { set_primitive(SD_CONE); });
    Node::bind("capsule", [&](const std::string& signal, void* button) { set_primitive(SD_CAPSULE); });
    Node::bind("cylinder", [&](const std::string& signal, void* button) { set_primitive(SD_CYLINDER); });
    Node::bind("torus", [&](const std::string& signal, void* button) { set_primitive(SD_TORUS); });
    Node::bind("vesica", [&](const std::string& signal, void* button) { set_primitive(SD_VESICA); });

    Node::bind("main_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(value); });
    Node::bind("secondary_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(-1.0f, value); });
    Node::bind("round_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(-1.0f, -1.0f, value); });

    //Node::bind("onion_value", [&](const std::string& signal, float value) { set_onion_modifier(value); });
    Node::bind("cap_value", (FuncFloat)[&](const std::string& signal, float value) { set_cap_modifier(value); });

    Node::bind("use_mirror", [&](const std::string& signal, void* button) { toggle_mirror(); });
    Node::bind("hide_mirror", [&](const std::string& signal, void* button) { hide_mirror = !hide_mirror; });
    Node::bind("mirror_translation", [&](const std::string& signal, void* button) { mirror_gizmo.set_operation(TRANSLATE); });
    Node::bind("mirror_rotation", [&](const std::string& signal, void* button) { mirror_gizmo.set_operation(ROTATE); });
    Node::bind("mirror_both", [&](const std::string& signal, void* button) { mirror_gizmo.set_operation(TRANSLATE | ROTATE); });
    Node::bind("snap_to_surface", [&](const std::string& signal, void* button) { snap_to_surface = !snap_to_surface; });
    Node::bind("snap_to_grid", [&](const std::string& signal, void* button) { snap_to_grid = !snap_to_grid; });
    Node::bind("lock_axis_toggle", [&](const std::string& signal, void* button) { axis_lock = !axis_lock; });
    Node::bind("lock_axis_x", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_X; });
    Node::bind("lock_axis_y", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Y; });
    Node::bind("lock_axis_z", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Z; });

    Node::bind("roughness", (FuncFloat)[&](const std::string& signal, float value) { stroke_parameters.set_material_roughness(value); });
    Node::bind("metallic", (FuncFloat)[&](const std::string& signal, float value) { stroke_parameters.set_material_metallic(value); });
    /*Node::bind("noise_intensity", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(value); });
    Node::bind("noise_frequency", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(-1.0f, value); });
    Node::bind("noise_octaves", [&](const std::string& signal, float value) { stroke_parameters.set_material_noise(-1.0f, -1.0f, static_cast<int>(value)); });
    Node::bind("noise_color_picker", [&](const std::string& signal, Color color) { stroke_parameters.set_material_noise_color(color); });*/

    Node::bind("color_picker", [&](const std::string& signal, Color color) { stroke_parameters.set_material_color(color); });
    Node::bind("pick_material", [&](const std::string& signal, void* button) { is_picking_material = !is_picking_material; });

    Node::bind("undo", [&](const std::string& signal, void* button) { undo(); });
    Node::bind("redo", [&](const std::string& signal, void* button) { redo(); });

    Node::bind("smooth_factor", (FuncFloat)[&](const std::string& signal, float value) { stroke_parameters.set_smooth_factor(value); });
    Node::bind("color_override", (FuncFloat)[&](const std::string& signal, float value) { stroke_parameters.set_color_override_factor(value); });

    // Bind colors callback...

    for (auto it : Node2D::all_widgets)
    {
        ui::Button2D* child = dynamic_cast<ui::Button2D*>(it.second);
        if (!child || !child->is_color_button) continue;
        Node::bind(child->get_name(), [&](const std::string& signal, void* button) {
            const Color& color = (reinterpret_cast<ui::Button2D*>(button))->color;
            stroke_parameters.set_material_color(color);
        });
    }

    // Bind recent color buttons...

    Node2D* color_picker = Node2D::get_widget_from_name("color_picker");
    max_recent_colors = color_picker->get_children().size();

    if (max_recent_colors > 0) {

        for (size_t i = 0; i < max_recent_colors; ++i)
        {
            ui::Button2D* child = static_cast<ui::Button2D*>(color_picker->get_children()[i]);
            Node::bind(child->get_name(), [&](const std::string& signal, void* button) {
                const Color& color = (reinterpret_cast<ui::Button2D*>(button))->color;
                stroke_parameters.set_material_color(color);
                Node::emit_signal("color_picker@changed", color);
            });
        }
    }
    else {
        spdlog::error("No recent_colors added!");
    }

    // Bind material samples callback...

    Node2D* samples_group = Node2D::get_widget_from_name("material_samples");
    if (samples_group) {
        for (size_t i = 0; i < samples_group->get_children().size(); ++i)
        {
            ui::Button2D* child = static_cast<ui::Button2D*>(samples_group->get_children()[i]);
            Node::bind(child->get_name(), [&](const std::string& signal, void* button) {
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
        Node2D* color_blending_modes = Node2D::get_widget_from_name("color_blend");
        if (color_blending_modes) {
            for (size_t i = 0; i < color_blending_modes->get_children().size(); ++i)
            {
                ui::Button2D* child = static_cast<ui::Button2D*>(color_blending_modes->get_children()[i]);
                Node::bind(child->get_name(), [&, index = i](const std::string& signal, void* button) {
                    stroke_parameters.set_color_blend_operation(static_cast<ColorBlendOp>(index));
                });
            }
        }
        else {
            spdlog::error("Cannot find color_blending_modes selector!");
        }
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

    Node2D* color_picker = Node2D::get_widget_from_name("color_picker");
    assert(color_picker);

    assert(recent_colors.size() <= color_picker->get_children().size());
    for (uint8_t i = 0; i < recent_colors.size(); ++i)
    {
        ui::Button2D* child = static_cast<ui::Button2D*>(color_picker->get_children()[i]);
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
    ui::ButtonSelector2D* mat_samples = static_cast<ui::ButtonSelector2D*>(Node2D::get_widget_from_name("material_samples"));
    assert(mat_samples);

    std::string name = "new_material_" + std::to_string(last_generated_material_uid++);

    auto callback = [&, p = mat_samples](const std::string& output) {

        ui::TextureButton2D* new_button = new ui::TextureButton2D(output, { "data/textures/material_samples.png", ui::UNIQUE_SELECTION });
        p->add_child(new_button);

        num_generated_materials++;

        // Add data to existing samples..
        const StrokeMaterial& mat = stroke_parameters.get_material();
        add_pbr_material_data(output, mat.color, mat.roughness, mat.metallic);
            // mat.noise_params.x, mat.noise_color, mat.noise_params.y, static_cast<int>(mat.noise_params.z));

        Node::bind(output, [&](const std::string& signal, void* button) {
            update_stroke_from_material(signal);
        });
    };

    ui::Keyboard::request(callback, name);
}

void SculptEditor::generate_random_material()
{
    // Set all data
    stroke_parameters.set_material_color(Color(random_f(), random_f(), random_f(), 1.0f));
    stroke_parameters.set_material_roughness(random_f());
    stroke_parameters.set_material_metallic(random_f());

    // Don't apply noise by now..
    //stroke_parameters.set_material_noise();

    update_gui_from_stroke_material(stroke_parameters.get_material());
}

void SculptEditor::update_gui_from_stroke_material(const StrokeMaterial& mat)
{
    // Emit signals to change UI values
    Node::emit_signal("color_picker@changed", mat.color);
    Node::emit_signal("roughness@changed", mat.roughness);
    Node::emit_signal("metallic@changed", mat.metallic);
    /*Node::emit_signal("noise_intensity@changed", mat.noise_params.x);
    Node::emit_signal("noise_frequency@changed", mat.noise_params.y);
    Node::emit_signal("noise_octaves@changed", mat.noise_params.z);
    Node::emit_signal("noise_color_picker@changed", mat.noise_color);*/
}

void SculptEditor::update_stroke_from_material(const std::string& name)
{
    const PBRMaterialData& data = pbr_materials_data[name];

    // Set all data
    stroke_parameters.set_material_color(data.base_color);
    stroke_parameters.set_material_roughness(data.roughness * 1.5f); // this is a hack because hdres don't have too much roughness..
    stroke_parameters.set_material_metallic(data.metallic);
    //stroke_parameters.set_material_noise(data.noise_params.x, data.noise_params.y, static_cast<int>(data.noise_params.z));

    update_gui_from_stroke_material(stroke_parameters.get_material());
}

void SculptEditor::pick_material()
{
    // Disable picking..
    if (is_picking_material) {
        Node::emit_signal("pick_material@pressed", (void*)nullptr);
        is_picking_material = false;
    }

    should_pick_material = false;

    const sGPU_RayIntersectionData& intersection = last_gpu_results.ray_intersection;

    if (intersection.has_intersected == 0u) {
        return;
    }

    // Set all data
    stroke_parameters.set_material_color(Color(intersection.intersection_albedo, 1.0f));
    stroke_parameters.set_material_roughness(intersection.intersection_roughness);
    stroke_parameters.set_material_metallic(intersection.intersection_metallic);
    // stroke_parameters.set_material_noise(-1.0f);

    update_gui_from_stroke_material(stroke_parameters.get_material());
}
