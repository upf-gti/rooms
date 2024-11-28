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

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/mesh_transparent.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

uint8_t SculptEditor::last_generated_material_uid = 0;

//Viewport3D* test_slider_thermometer = nullptr;

uint8_t get_leading_thumbstick_axis(const glm::vec2 thumb_axis);

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    sSDFGlobals& sdf_globals = renderer->get_sdf_globals();

    stroke_manager.set_brick_world_size(glm::vec3(sdf_globals.brick_world_size));

    mirror_mesh = new MeshInstance3D();
    mirror_mesh->add_surface(RendererStorage::get_surface("quad"));
    mirror_mesh->scale(glm::vec3(0.25f));

    Material* mirror_material = new Material();
    mirror_material->set_priority(0);
    mirror_material->set_transparency_type(ALPHA_BLEND);
    mirror_material->set_cull_type(CULL_NONE);
    mirror_material->set_diffuse_texture(RendererStorage::get_texture("data/textures/mirror_quad_texture.png"));
    mirror_material->set_type(MATERIAL_UNLIT);
    mirror_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, mirror_material));

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

        sculpt_area_box->set_surface_material_override(s, ref_mat);
        sculpt_area_box->add_surface(s);
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
        preview_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_transparent::source, shaders::mesh_transparent::path, preview_material));

        mesh_preview->set_surface_material_override(sphere_surface, preview_material);

        mesh_preview_outline = new MeshInstance3D();
        mesh_preview_outline->add_surface(sphere_surface);

        Material* outline_material = new Material();
        outline_material->set_cull_type(CULL_FRONT);
        outline_material->set_type(MATERIAL_UNLIT);
        outline_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, outline_material));

        mesh_preview_outline->set_surface_material_override(sphere_surface, outline_material);
    }

    // Default stroke config
    {
        stroke_parameters.set_primitive(SD_SPHERE);
        stroke_parameters.set_operation(OP_SMOOTH_UNION);
        stroke_parameters.set_color_blend_operation(COLOR_OP_REPLACE);
        stroke_parameters.set_parameters({ 0.0f, -1.0f, 0.0f, 0.005f });
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

        SculptManager::sGPU_ReadResults* gpu_result = reinterpret_cast<SculptManager::sGPU_ReadResults*>(data);
        assert(gpu_result);

        last_gpu_results = gpu_result->loaded_results;

        // Computing sculpt AABB
        {
            const sGPU_SculptResults::sGPU_SculptEvalData& eval_data = last_gpu_results.sculpt_eval_data;

            // If there has been an eval, assign the AABB to the sculpt
            if (static_cast<RoomsRenderer*>(engine->get_renderer())->has_performed_evaluation()) {
                glm::vec3 half_size = (eval_data.aabb_max - eval_data.aabb_min) / 2.0f;
                AABB result = { result.half_size + eval_data.aabb_min, half_size };
                result = result.transform(Transform::transform_to_mat4(get_current_transform()));
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

    /*ui::VContainer2D* test_root = new ui::VContainer2D("test_root", { 0.0f, 0.0f });
    test_root->set_centered(true);

    test_root->add_child(new ui::Slider2D("thermometer", 0.0f, ui::SliderMode::HORIZONTAL, ui::DISABLED));*/

    /*test_slider_thermometer = new Viewport3D(test_root);
    test_slider_thermometer->set_active(true);

    RoomsEngine::entities.push_back(test_slider_thermometer);*/
}

void SculptEditor::on_enter(void* data)
{
    SculptNode* sculpt_node = reinterpret_cast<SculptNode*>(data);
    assert(sculpt_node);
    set_current_sculpt(sculpt_node);

    /*
    * If loaded from memory, we can assume it has a defined position,
    * so do not move it and start now the sculpt.
    */

    if (!sculpt_started && sculpt_node->get_from_memory()) {
        sculpt_started = true;
    }

    if (sculpt_started) {
        if (sculpt_node->get_parent()) {
            current_instance_transform = sculpt_node->get_parent<Node3D*>()->get_transform();
        }
        else {
            current_instance_transform = sculpt_node->get_transform();
        }


    }

    // Store if we started from scratch the sculpt to assign or not its new position
    sculpt_from_zero = !sculpt_started;

    static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->set_preview_render(true);

    update_ui_workflow_state();
}

void SculptEditor::on_exit()
{
    if (sculpt_from_zero) {
        current_sculpt->set_transform(current_instance_transform);
    }

    static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->set_preview_render(false);
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

bool SculptEditor::is_out_of_focus(SculptNode* sculpt_node)
{
    if (!renderer->get_openxr_available()) {
        return get_current_sculpt() != sculpt_node;
    }

    return true;
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

    if (renderer->get_openxr_available()) {
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
            tmp.color = Color(intersection.intersection_albedo, 1.0f);
            tmp.roughness = intersection.intersection_roughness;
            tmp.metallic = intersection.intersection_metallic;
            update_gui_from_stroke_material(tmp);
        }

        if (is_tool_used) {
            pick_material();
            return false;
        }
    }

    // Move the edit a little away
    glm::mat4x4 controller_pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
    controller_pose = glm::translate(controller_pose, glm::vec3(0.0f, 0.0f, -hand_to_edit_distance));

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

    // Guides: edit position modifiers
    if(renderer->get_openxr_available()) {
        // Snap surface

        if (snap_to_surface) {

            test_ray_to_sculpts();

            if (can_snap_to_surface()) {
                edit_to_add.position = last_snap_position;
            }
        }

        if (use_mirror) {
            bool r = mirror_gizmo.update(Input::get_controller_position(HAND_RIGHT, POSE_AIM), delta_time);
            is_tool_used &= !r;
            is_tool_pressed &= !r;
            mirror_normal = glm::normalize(mirror_gizmo.get_rotation() * glm::vec3(0.f, 0.f, 1.f));
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

    // Update edit dimensions
    if(!is_stretching_edit) {

        // Get the data from the primitive default
        edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;

        const glm::vec2 thumbstick_values = Input::get_thumbstick_value(HAND_RIGHT);

        // Disable the unused joystick axis until the joystick is released
        uint8_t curr_thumbstick_axis = get_leading_thumbstick_axis(thumbstick_values);

        if (thumbstick_leading_axis == THUMBSTICK_NO_AXIS) {
            thumbstick_leading_axis = curr_thumbstick_axis;
        } else if (curr_thumbstick_axis == THUMBSTICK_NO_AXIS) {
            thumbstick_leading_axis = THUMBSTICK_NO_AXIS;
        }

        bool use_x_axis = thumbstick_leading_axis == THUMBSTICK_AXIS_X;

        float size_multiplier = (use_x_axis ? thumbstick_values.x : thumbstick_values.y * 0.1f) * delta_time;

        if (std::abs(size_multiplier) > 0.f) {
            // Update primitive main size
            // TODO: When smearing, always change main size (using y) by now!
            if (!is_shift_right_pressed || (is_tool_pressed && !use_x_axis)) {
                if (use_x_axis) {
                    // Update rounded size
                    edit_to_add.dimensions.w = glm::clamp(edit_to_add.dimensions.w + size_multiplier * 0.1f, 0.0f, MAX_PRIMITIVE_SIZE);
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
                    stroke_parameters.set_smooth_factor(glm::clamp(current_smooth + size_multiplier * 0.02f, MIN_SMOOTH_FACTOR, MAX_SMOOTH_FACTOR));
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

        edit_to_add.rotation = glm::inverse(Input::get_controller_rotation(HAND_RIGHT, POSE_AIM));
    }

    // Stretch the edit using motion controls
    if (stamp_enabled && is_tool_pressed && !creating_spline) {

        if (is_stretching_edit) {
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
        else {
            // Only stretch the edit when the acceleration of the hand exceds a threshold
            is_stretching_edit = glm::length(glm::abs(controller_movement_data[HAND_RIGHT].velocity)) > 0.20f;
        }
    }

    if (!creating_spline) {
        update_edit_rotation();
    }

    // Debug sculpting
    {
        // For debugging sculpture without a headset
        if (!renderer->get_openxr_available()) {

            if (is_tool_being_used(stamp_enabled)) {
                edit_to_add.position = get_current_transform().get_position() + glm::vec3(glm::vec3(0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1), 0.2f * (random_f() * 2 - 1)));
                glm::vec3 euler_angles(glm::pi<float>() * random_f(), glm::pi<float>()* random_f(), glm::pi<float>()* random_f());
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
                edit_to_add.position = get_current_transform().get_position();
                edit_to_add.rotation = get_current_transform().get_rotation();
            }
        }
    }

    // Store now since later it will be converted to 3d texture space
    edit_position_world = edit_to_add.position;
    edit_rotation_world = edit_to_add.rotation;

    // Add edit based on controller movement
    // TODO(Juan): Check rotation?
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

    return is_tool_used;
}

void SculptEditor::update(float delta_time)
{
    /*glm::mat4x4 m(1.0f);

    glm::vec3 eye = renderer->get_camera_eye();
    glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.5f;

    m = glm::translate(m, new_pos);
    m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));

    test_slider_thermometer->set_model(m);*/

    if (current_tool == NONE) {
        return;
    }

    BaseEditor::update(delta_time);

    // Update controller UI
    if (renderer->get_openxr_available()) {
        generate_shortcuts();
    }

    preview_tmp_edits.clear();
    new_edits.clear();

    // Operation changer for the different tools
    {
        if (Input::was_button_pressed(XR_BUTTON_B)) {

            if (creating_spline) {
                reset_spline();
            }
            else if (is_shift_right_pressed) {
                snap_to_surface = !snap_to_surface;
            }
            else {
                start_spline();
            }
        }

        if (Input::was_button_pressed(XR_BUTTON_A)) {

            if (creating_spline) {
                end_spline();
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

    if (Input::was_button_pressed(XR_BUTTON_Y) && !creating_spline) {
        RoomsEngine::switch_editor(SCENE_EDITOR);
    }

    bool is_tool_used = edit_update(delta_time);

    // Sculpt lifecicle
    {
        // Set center of sculpture and reuse it as mirror center
        if (!sculpt_started) {

            const glm::vec3& position = edit_to_add.position;

            if (renderer->get_openxr_available()) {
                get_current_transform().set_position(position);
            }

            Transform& mirror_transform = mirror_gizmo.get_transform();
            mirror_transform.set_position(position);
            Transform& lock_axis_transform = axis_lock_gizmo.get_transform();
            lock_axis_transform.set_position(position);
        }

        // Mark the start of the sculpture for the origin
        if (current_tool == SCULPT && is_tool_used) {
            sculpt_started = true;
        }
    }

    update_sculpt_rotation();

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

        // Upload the edit to the  edit list
        if (is_tool_used) {

            // Manage splines
            if (creating_spline && stamp_enabled) {
                current_spline.add_knot( { edit_to_add.position, edit_to_add.dimensions } );

                if (current_spline.size() >= MAX_KNOTS_PER_SPLINE) {
                    end_spline();
                }
            }
            else {
                new_edits.push_back(edit_to_add);

                // a hack for flatscreen sculpting
                if (!renderer->get_openxr_available() && new_edits.size() > 0u) {
                    stroke_manager.change_stroke_params(stroke_parameters);
                }
            }

            // Add recent color only when is used...
            add_recent_color(stroke_parameters.get_material().color);
        }

        // Submit spline edits in the next frame..
        else if (dirty_spline) {

            current_spline.for_each([&](const Knot& point) {
                Edit edit;
                edit.position = point.position;
                edit.dimensions = glm::vec4(point.size, 0.0f);
                new_edits.push_back(edit);
            });

            force_new_stroke = true;
            reset_spline();
        }
    }

    // Add preview edits

    if (creating_spline) {

        preview_spline = current_spline;

        preview_spline.add_knot({ edit_to_add.position, edit_to_add.dimensions });

        preview_spline.for_each([&](const Knot& point) {
            Edit edit;
            edit.position = point.position;
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

    set_preview_edits(preview_tmp_edits);

    bool needs_evaluation = false;
    if (called_undo) {
        needs_evaluation = stroke_manager.undo();
        called_undo = false;
    } else if (called_redo) {
        needs_evaluation = stroke_manager.redo();
        called_redo = false;
    } else if (new_edits.size() > 0u) {
        needs_evaluation = stroke_manager.add(new_edits);
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
    if (needs_evaluation || is_released || (is_tool_used && !renderer->get_openxr_available())) {
        update_ui_workflow_state();
    }
    
    was_tool_used = is_tool_used;

    // Render current instance
    if(renderer->get_openxr_available()) {
        uint32_t flags = 0u;
        RoomsRenderer* renderer = static_cast<RoomsRenderer*>(Renderer::instance);
        in_frame_sculpt_render_list_id = renderer->add_sculpt_render_call(
            current_sculpt->get_sculpt_data(), Transform::transform_to_mat4(get_current_transform()), flags);
        in_frame_sculpt_render_list_id += current_sculpt->get_sculpt_data()->get_in_frame_model_buffer_index();
    }

    if (is_tool_used) {
        renderer->toogle_frame_debug();

        if (is_stretching_edit) {
            is_stretching_edit = false;
            edit_to_add.dimensions = primitive_default_states[stroke_parameters.get_primitive()].dimensions;
            dimensions_dirty = true;
        }

        /*renderer->get_raymarching_renderer()->get_brick_usage([](float pct, uint32_t brick_count) {
            Node::emit_signal("thermometer@changed", pct);
        });*/
    }
}

void SculptEditor::set_preview_edits(const std::vector<Edit>& edit_previews)
{
    sGPUStroke preview_stroke;

    preview_stroke.color_blending_op = stroke_parameters.get_color_blend_operation();
    preview_stroke.primitive = stroke_parameters.get_primitive();
    preview_stroke.material = stroke_parameters.get_material();
    preview_stroke.operation = stroke_parameters.get_operation();
    preview_stroke.parameters = stroke_parameters.get_parameters();

    preview_stroke.edit_count = preview_tmp_edits.size();

    AABB stroke_aabb = preview_stroke.get_world_AABB_of_edit_list(edit_previews);
    preview_stroke.aabb_min = stroke_aabb.center - stroke_aabb.half_size;
    preview_stroke.aabb_max = stroke_aabb.center + stroke_aabb.half_size;

    renderer->get_sculpt_manager()->set_preview_stroke(
        current_sculpt->get_sculpt_data(),
        renderer->get_openxr_available() ? in_frame_sculpt_render_list_id : current_sculpt->get_in_frame_model_idx(),
        preview_stroke, edit_previews
    );
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
        pos_texture_space -= (get_current_transform().get_position());
    }

    pos_texture_space = glm::inverse(get_current_transform().get_rotation()) * pos_texture_space;

    return pos_texture_space;
}

glm::vec3 SculptEditor::texture3d_to_world(const glm::vec3& position)
{
    glm::vec3 pos_world_space;

    pos_world_space = glm::inverse(get_current_transform().get_rotation()) * position;
    pos_world_space = pos_world_space + (get_current_transform().get_position());

    return pos_world_space;
}

void SculptEditor::test_ray_to_sculpts()
{
    // Send rays each frame to detect hovered sculpts and other nodes
    Engine::instance->get_scene_ray(ray_origin, ray_direction);

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    rooms_renderer->get_sculpt_manager()->set_ray_to_test(ray_origin, ray_direction, current_sculpt->get_sculpt_data(), in_frame_sculpt_render_list_id);
}

void SculptEditor::update_sculpt_rotation()
{
    if (Input::was_key_pressed(GLFW_KEY_R)) {
        get_current_transform().translate(glm::vec3(0.5, 0.0f, 0.0f));
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

        get_current_transform().rotate_world(rotation_diff);
        get_current_transform().translate(translation_diff);

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
    edit_to_add.rotation *= (get_current_transform().get_rotation());
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

        if (creating_spline) {
            can_undo = (current_spline.size() > 0u);
        }

        auto b_undo = static_cast<ui::Button2D*>(Node2D::get_widget_from_name("undo"));
        b_undo->set_disabled(!can_undo);

        // TODO: Spline knot cannot be redone by now!
        bool can_redo = stroke_manager.can_redo() && !creating_spline;
        auto b_redo = static_cast<ui::Button2D*>(Node2D::get_widget_from_name("redo"));
        b_redo->set_disabled(!can_redo);
    }
}

void SculptEditor::undo()
{
    if (creating_spline) {
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
    if (creating_spline) {
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
        if (renderer->get_openxr_available()) {

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
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        else if (axis_lock_mode & AXIS_LOCK_Y)
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        mirror_mesh->render();
    }
    else if (use_mirror) {

        mirror_gizmo.render();

        if (!renderer->get_openxr_available()) {
            mirror_normal = glm::normalize(mirror_gizmo.get_rotation() * glm::vec3(0.f, 0.f, 1.f));
        }

        mirror_mesh->set_transform(mirror_gizmo.get_transform());
        mirror_mesh->scale(glm::vec3(0.5f));
        mirror_mesh->render();
    }

    BaseEditor::render();

    RoomsEngine::render_controllers();

    // Render always or only XR?
    sculpt_area_box->set_transform(Transform::identity());
    sculpt_area_box->translate(get_current_transform().get_position());
    sculpt_area_box->scale(glm::vec3(SCULPT_MAX_SIZE * 0.5f));
    sculpt_area_box->rotate(get_current_transform().get_rotation());
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

void SculptEditor::set_sculpt_started(bool value)
{
    sculpt_started = true;
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
        hand_to_edit_distance = 0.05f;
        static_cast<ui::ButtonSubmenu2D*>(brush_editor_submenu)->set_disabled(true);
        Node::emit_signal("add@pressed", (void*)nullptr);
        break;
    case PAINT:
        stroke_parameters.set_operation(OP_SMOOTH_PAINT);
        hand_to_edit_distance = 0.15f;
        static_cast<ui::ButtonSubmenu2D*>(brush_editor_submenu)->set_disabled(false);
        Node::emit_signal("paint@pressed", (void*)nullptr);
        break;
    default:
        break;
    }

    // Set this to allow the mesh preview to give a little mergin in the outline mode
    dimensions_dirty = true;
}

/*
*   In 2D, current_instance_transform is not used,
*   so use always the transform of the current sculpt
*/
Transform& SculptEditor::get_current_transform()
{
    if (!renderer->get_openxr_available()) {
        return current_sculpt->get_transform();
    }

    return current_instance_transform;
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
    if (creating_spline) {
        reset_spline(false);
    }
    else {
        creating_spline = true;
        current_spline.clear();

        if (update_ui) {
            Node::emit_signal("create_spline@pressed", (void*)nullptr);
        }
    }
}

void SculptEditor::reset_spline(bool update_ui)
{
    dirty_spline = false;
    creating_spline = false;
    current_spline.clear();

    if (update_ui) {
        Node::emit_signal("create_spline@pressed", (void*)nullptr);
    }
}

void SculptEditor::end_spline()
{
    dirty_spline = true;

    update_ui_workflow_state();
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

    if (creating_spline) {
        shortcuts[shortcuts::ADD_KNOT] = true;
        shortcuts[shortcuts::CONFIRM_SPLINE] = true;
        shortcuts[shortcuts::CANCEL_SPLINE] = true;
    }
    else {
        shortcuts[shortcuts::ADD_SPLINE] = !is_shift_right_pressed;
        shortcuts[shortcuts::BACK_TO_SCENE] = true;
        shortcuts[shortcuts::MAIN_SIZE] = !is_shift_right_pressed;
        shortcuts[shortcuts::SECONDARY_SIZE] = is_shift_right_pressed;
        shortcuts[shortcuts::ADD_SUBSTRACT] = !is_shift_right_pressed;
        shortcuts[shortcuts::ROUND_SHAPE] = !is_shift_right_pressed;
        shortcuts[shortcuts::SNAP_SURFACE] = is_shift_right_pressed;
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

    main_panel = new ui::HContainer2D("root", { 48.0f, screen_size.y - 224.f }, ui::CREATE_3D);

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

    // DEBUG SPLINES: REMOVE THIS!!
    first_row->add_child(new ui::TextureButton2D("create_spline", { "data/textures/bezier.png", ui::ALLOW_TOGGLE }));

    // ** Shape, Brush, Material Editors **
    {
        ui::ItemGroup2D* g_editors = new ui::ItemGroup2D("g_editors");

        // Shape editor
        {
            ui::ButtonSubmenu2D* shape_editor_submenu = new ui::ButtonSubmenu2D("shape_editor", { "data/textures/shape_editor.png" });

            // Edit sizes
            {
                ui::ItemGroup2D* g_edit_sizes = new ui::ItemGroup2D("g_edit_sizes");
                g_edit_sizes->add_child(new ui::FloatSlider2D("main_size", edit_to_add.dimensions.x, ui::SliderMode::HORIZONTAL, 0, MIN_PRIMITIVE_SIZE, MAX_PRIMITIVE_SIZE, 3));
                g_edit_sizes->add_child(new ui::FloatSlider2D("secondary_size", edit_to_add.dimensions.y, ui::SliderMode::HORIZONTAL, 0, MIN_PRIMITIVE_SIZE, MAX_PRIMITIVE_SIZE, 3));
                g_edit_sizes->add_child(new ui::FloatSlider2D("round_size", "data/textures/rounding.png", edit_to_add.dimensions.w, ui::SliderMode::VERTICAL, 0, 0.0f, MAX_PRIMITIVE_SIZE, 2));
                shape_editor_submenu->add_child(g_edit_sizes);
            }

            // Edit modifiers
            {
                ui::ItemGroup2D* g_edit_modifiers = new ui::ItemGroup2D("g_edit_modifiers");
                //g_edit_modifiers->add_child(new ui::FloatSlider2D("onion_value", "data/textures/onion.png", 0.0f, ui::SliderMode::VERTICAL));
                g_edit_modifiers->add_child(new ui::FloatSlider2D("cap_value", "data/textures/capped.png", 0.0f, ui::SliderMode::VERTICAL));
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
            g_edit_pbr->add_child(new ui::FloatSlider2D("roughness", "data/textures/roughness.png", stroke_material.roughness));
            g_edit_pbr->add_child(new ui::FloatSlider2D("metallic", "data/textures/metallic.png", stroke_material.metallic));
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
            mirror_submenu->add_child(new ui::TextureButton2D("mirror_toggle", { "data/textures/mirror.png", ui::ALLOW_TOGGLE }));
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

    // ** Undo/Redo **
    {
        second_row->add_child(new ui::TextureButton2D("undo", { "data/textures/undo.png" }));
        second_row->add_child(new ui::TextureButton2D("redo", { "data/textures/redo.png" }));
    }

    // Smooth factor
    {
        ui::Slider2D* smooth_factor_slider = new ui::FloatSlider2D("smooth_factor", "data/textures/smooth.png", stroke_parameters.get_smooth_factor(), ui::SliderMode::VERTICAL, ui::SKIP_VALUE, MIN_SMOOTH_FACTOR, MAX_SMOOTH_FACTOR, 3);
        second_row->add_child(smooth_factor_slider);
    }

    // Load controller UI labels
    if (renderer->get_openxr_available())
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
            right_hand_box->add_child(new ui::ImageLabel2D("Main size", shortcuts::R_THUMBSTICK_PATH, shortcuts::MAIN_SIZE));
            right_hand_box->add_child(new ui::ImageLabel2D("Sec size", shortcuts::R_GRIP_R_THUMBSTICK_PATH, shortcuts::SECONDARY_SIZE, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Round Shape", shortcuts::R_THUMBSTICK_PATH, shortcuts::ROUND_SHAPE));
            right_hand_box->add_child(new ui::ImageLabel2D("Smooth", shortcuts::R_GRIP_R_THUMBSTICK_PATH, shortcuts::MODIFY_SMOOTH, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Add Spline", shortcuts::B_BUTTON_PATH, shortcuts::ADD_SPLINE));
            right_hand_box->add_child(new ui::ImageLabel2D("Cancel Spline", shortcuts::B_BUTTON_PATH, shortcuts::CANCEL_SPLINE));
            right_hand_box->add_child(new ui::ImageLabel2D("Surface Snap", shortcuts::R_GRIP_B_BUTTON_PATH, shortcuts::SNAP_SURFACE, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Confirm Spline", shortcuts::A_BUTTON_PATH, shortcuts::CONFIRM_SPLINE));
            right_hand_box->add_child(new ui::ImageLabel2D("Add/Substract", shortcuts::A_BUTTON_PATH, shortcuts::ADD_SUBSTRACT));
            right_hand_box->add_child(new ui::ImageLabel2D("Pick Material", shortcuts::R_GRIP_A_BUTTON_PATH, shortcuts::PICK_MATERIAL, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Stamp", shortcuts::R_TRIGGER_PATH, shortcuts::STAMP));
            right_hand_box->add_child(new ui::ImageLabel2D("Add Knot", shortcuts::R_TRIGGER_PATH, shortcuts::ADD_KNOT));
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

    Node::bind("create_spline", [&](const std::string& signal, void* button) { start_spline(false); });

    Node::bind("main_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(value); });
    Node::bind("secondary_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(-1.0f, value); });
    Node::bind("round_size", (FuncFloat)[&](const std::string& signal, float value) { set_edit_size(-1.0f, -1.0f, value); });

    //Node::bind("onion_value", [&](const std::string& signal, float value) { set_onion_modifier(value); });
    Node::bind("cap_value", (FuncFloat)[&](const std::string& signal, float value) { set_cap_modifier(value); });

    Node::bind("mirror_toggle", [&](const std::string& signal, void* button) { use_mirror = !use_mirror; });
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
    const sGPU_RayIntersectionData& intersection = last_gpu_results.ray_intersection;

    // Do not check _has_intersected_ since we can assume we triggered the action..

    // Set all data
    stroke_parameters.set_material_color(Color(intersection.intersection_albedo, 1.0f));
    stroke_parameters.set_material_roughness(intersection.intersection_roughness);
    stroke_parameters.set_material_metallic(intersection.intersection_metallic);
    // stroke_parameters.set_material_noise(-1.0f);

    // Disable picking..
    if (is_picking_material) {
        Node::emit_signal("pick_material@pressed", (void*)nullptr);
        is_picking_material = false;
    }

    should_pick_material = false;

    update_gui_from_stroke_material(stroke_parameters.get_material());
}


uint8_t get_leading_thumbstick_axis(const glm::vec2 thumb_axis) {
    const glm::vec2 abs_axis = glm::abs(thumb_axis);
    if (glm::abs(glm::length(abs_axis)) >= THUMBSTICK_DEADZONE) {
        if (abs_axis.x > abs_axis.y) {
            return THUMBSTICK_AXIS_X;
        } else {
            return THUMBSTICK_AXIS_Y;
        }
    }

    return THUMBSTICK_NO_AXIS;
}
