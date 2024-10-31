#include "group_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/parsers/parse_scene.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/group_3d.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/math/intersections.h"
#include "framework/ui/inspector.h"
#include "framework/math/math_utils.h"
#include "framework/camera/camera.h"
#include "framework/resources/sculpt.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "graphics/managers/sculpt_manager.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

uint64_t GroupEditor::node_signal_uid = 0;

void GroupEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    gizmo = static_cast<RoomsEngine*>(RoomsEngine::instance)->get_gizmo();

    init_ui();

    Node::bind("@on_gpu_results", [&](const std::string& sg, void* data) {

        // Do nothing if it's not the current editor..
        auto engine = static_cast<RoomsEngine*>(RoomsEngine::instance);;
        if (engine->get_current_editor() != this) {
            return;
        }

        hovered_node = nullptr;

        SculptManager::sGPU_ReadResults* gpu_result = reinterpret_cast<SculptManager::sGPU_ReadResults*>(data);
        assert(gpu_result);
        const sGPU_SculptResults::sGPU_IntersectionData& intersection = gpu_result->loaded_results.ray_intersection;

        if (intersection.has_intersected == 0u) {
            return;
        }

        auto& nodes = current_group->get_children();
        for (auto node : nodes) {
            SculptNode* sculpt_node = dynamic_cast<SculptNode*>(node);
            if (sculpt_node && sculpt_node->check_intersection(intersection.sculpt_id, intersection.instance_id)) {
                hovered_node = sculpt_node;
                break;
            }
        }
    });
}

void GroupEditor::clean()
{
    BaseEditor::clean();
}

void GroupEditor::on_enter(void* data)
{
    current_group = reinterpret_cast<Group3D*>(data);
    assert(current_group);

    gizmo->set_operation(TRANSLATE);
    Node::emit_signal("combo_gizmo_modes@changed", (void*)"translate");
}

void GroupEditor::update(float delta_time)
{
    BaseEditor::update(delta_time);

    // Update input actions
    {
        select_action_pressed = Input::was_trigger_pressed(HAND_RIGHT) || Input::was_mouse_pressed(GLFW_MOUSE_BUTTON_LEFT);
    }

    update_hovered_node();

    shortcuts.clear();
    shortcuts[shortcuts::TOGGLE_SCENE_INSPECTOR] = true;

    if (hovered_node) {
        process_node_hovered();
    }
    else {
        shortcuts[shortcuts::SELECT_NODE] = true;
    }

    if (inspector_dirty) {
        inspect_group();
    }

    if (Input::was_button_pressed(XR_BUTTON_Y) || Input::was_key_pressed(GLFW_KEY_I)) {
        inspect_group(true);
    }

    update_gizmo(delta_time);

    update_node_transform();

    if (renderer->get_openxr_available()) {

        if (inspector_transform_dirty) {
            update_panel_transform();
        }

        BaseEditor::update_shortcuts(shortcuts);
    }

    inspector->update(delta_time);
}

void GroupEditor::render()
{
    RoomsEngine::render_controllers();

    render_gizmo();

    BaseEditor::render();

    inspector->render();
}

void GroupEditor::render_gui()
{
    if (selected_node) {
        ImGui::Text("Selected Node: %s", selected_node->get_name().c_str());
    }

    if (hovered_node) {

        ImGui::Text("Hovered Node");

        //ImGui::OpenPopup("context_menu_popup");

        //if (ImGui::BeginPopup("context_menu_popup")) {

        //    if (ImGui::MenuItem("Clone")) {
        //        // Handle clone action..
        //    }
        //    if (ImGui::MenuItem("Delete")) {
        //        // Handle delete action..
        //    }

        //    ImGui::EndPopup();
        //}
    }
}

void GroupEditor::update_hovered_node()
{
    // Send rays each frame to detect hovered sculpts and other nodes

    glm::vec3 ray_origin;
    glm::vec3 ray_direction;

    if (Renderer::instance->get_openxr_available()) {
        ray_origin = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
        glm::mat4x4 select_hand_pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        ray_direction = get_front(select_hand_pose);
    }
    else {
        Camera* camera = Renderer::instance->get_camera();
        glm::vec3 ray_dir = camera->screen_to_ray(Input::get_mouse_position());
        ray_origin = camera->get_eye();
        ray_direction = glm::normalize(ray_dir);
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    rooms_renderer->get_sculpt_manager()->set_ray_to_test(ray_origin, ray_direction);
}

void GroupEditor::process_node_hovered()
{
    const bool a_pressed = Input::was_button_pressed(XR_BUTTON_A);
    const bool b_pressed = Input::was_button_pressed(XR_BUTTON_B);

    if (a_pressed) {
        ungroup_node(hovered_node);
    }
    else if (b_pressed) {
        RoomsEngine::switch_editor(SCENE_EDITOR);
    }
    else if (select_action_pressed) {
        select_node(hovered_node, false);
    }
}

void GroupEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel = new ui::HContainer2D("group_editor_root", { 48.0f, screen_size.y - 200.f }, ui::CREATE_3D);

    ui::VContainer2D* vertical_container = new ui::VContainer2D("scene_vertical_container", { 0.0f, 0.0f });
    main_panel->add_child(vertical_container);

    // Add main rows
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    ui::HContainer2D* second_row = new ui::HContainer2D("row_1", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    first_row->add_child(new ui::TextureButton2D("deselect", "data/textures/cross.png"));
    first_row->add_child(new ui::TextureButton2D("ungroup", "data/textures/ungroup.png"));

    // ** Go back to scene editor **
    second_row->add_child(new ui::TextureButton2D("go_back", "data/textures/back.png"));

    // ** Gizmo modes **
    {
        ui::ComboButtons2D* combo_gizmo_modes = new ui::ComboButtons2D("combo_gizmo_modes");
        combo_gizmo_modes->add_child(new ui::TextureButton2D("no_gizmo", "data/textures/no_gizmo.png"));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("translate", "data/textures/translation_gizmo.png", ui::SELECTED));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("rotate", "data/textures/rotation_gizmo.png"));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("scale", "data/textures/scale_gizmo.png"));
        second_row->add_child(combo_gizmo_modes);
    }

    // Create inspection panel (Nodes, properties, etc)
    {
        inspector = new ui::Inspector({ .name = "inspector_root", .title = "Group Nodes",.position = {32.0f, 32.f} });
        inspector->set_visibility(false);
    }

    if (renderer->get_openxr_available())
    {
        // Load controller UI labels

        // Thumbsticks
        // Buttons
        // Triggers

        glm::vec2 double_size = { 2.0f, 1.0f };

        // Left hand
        {
            left_hand_box = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            left_hand_box->add_child(new ui::ImageLabel2D("Scene Panel", shortcuts::Y_BUTTON_PATH, shortcuts::TOGGLE_SCENE_INSPECTOR));
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            right_hand_box->add_child(new ui::ImageLabel2D("Back to scene", shortcuts::B_BUTTON_PATH, shortcuts::BACK_TO_SCENE));
            right_hand_box->add_child(new ui::ImageLabel2D("Ungroup", shortcuts::A_BUTTON_PATH, shortcuts::UNGROUP));
            right_hand_box->add_child(new ui::ImageLabel2D("Select Node", shortcuts::R_TRIGGER_PATH, shortcuts::SELECT_NODE));
        }
    }

    // Bind callbacks
    bind_events();
}

void GroupEditor::bind_events()
{
    Node::bind("deselect", [&](const std::string& signal, void* button) { deselect(); });
    Node::bind("ungroup", [&](const std::string& signal, void* button) { ungroup_node(selected_node); });
}

void GroupEditor::select_node(Node* node, bool place)
{
    selected_node = node;

    // Update inspector selection
    Node::emit_signal(node->get_name() + "_label@selected", (void*)nullptr);
}

void GroupEditor::deselect()
{
    // hack by now: Do nothing if it's not the current editor..
    auto engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
    if (engine->get_current_editor() != this) {
        return;
    }

    selected_node = nullptr;

    ui::Text2D::select(nullptr);
}

void GroupEditor::ungroup_node(Node* node)
{
    if (!node) {
        return;
    }

    if (selected_node == node) {
        deselect();
    }

    // Store world transform
    glm::mat4x4 world_space_model = static_cast<Node3D*>(node)->get_global_model();

    current_group->remove_child(node);

    // Add back to the scene
    Scene* main_scene = Engine::instance->get_main_scene();
    main_scene->add_node(node);

    // Offset the node so it keeps its world position even without the parent
    static_cast<Node3D*>(node)->set_transform(Transform::mat4_to_transform(world_space_model));

    inspector_dirty = true;

    // Delete group and go back to scene editor in case no nodes left in group
    if (current_group->get_children().empty()) {
        main_scene->remove_node(current_group);
        delete current_group;
        current_group = nullptr;

        RoomsEngine::switch_editor(SCENE_EDITOR);
    }
}

bool GroupEditor::is_gizmo_usable()
{
    bool r = !!selected_node;

    if (r) {
        r &= !!dynamic_cast<Node3D*>(selected_node);
    }

    return r;
}

void GroupEditor::update_gizmo(float delta_time)
{
    if (!is_gizmo_usable()) {
        return;
    }

    // Only 3D Gizmo for XR needs to update

    if (!renderer->get_openxr_available()) {
        return;
    }

    // Make sure to set gizmo in node space

    Node3D* node = static_cast<Node3D*>(selected_node);
    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);

    const Transform& parent_transform = node->get_parent<Node3D*>()->get_transform();
    Transform t = Transform::combine(parent_transform, node->get_transform());

    if (gizmo->update(t, right_controller_pos, delta_time)) {
        node->set_transform(Transform::combine(Transform::inverse(parent_transform), t));
    }
}

void GroupEditor::render_gizmo()
{
    if (!is_gizmo_usable()) {
        return;
    }

    // Make sure to set gizmo in node space

    Node3D* node = static_cast<Node3D*>(selected_node);

    const Transform& parent_transform = node->get_parent<Node3D*>()->get_transform();
    gizmo->set_transform(Transform::combine(parent_transform, node->get_transform()));

    bool transform_dirty = gizmo->render();

    if (transform_dirty) {
        node->set_transform(Transform::combine(Transform::inverse(parent_transform), gizmo->get_transform()));
    }
}

void GroupEditor::update_panel_transform()
{
    glm::mat4x4 m(1.0f);
    glm::vec3 eye = renderer->get_camera_eye();
    glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.6f;

    m = glm::translate(m, new_pos);
    m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
    m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

    inspector->set_xr_transform(Transform::mat4_to_transform(m));

    inspector_transform_dirty = false;
}

bool GroupEditor::is_rotation_being_used()
{
    return Input::get_trigger_value(HAND_LEFT) > 0.5;
}

void GroupEditor::update_node_transform()
{
    // Do not rotate sculpt if shift -> we might be rotating the edit
    if (selected_node && is_rotation_being_used() && !is_shift_left_pressed) {

        glm::quat left_hand_rotation = Input::get_controller_rotation(HAND_LEFT);
        glm::vec3 left_hand_translation = Input::get_controller_position(HAND_LEFT);
        glm::vec3 right_hand_translation = Input::get_controller_position(HAND_RIGHT);
        float hand_distance = glm::length2(left_hand_translation - right_hand_translation);

        if (!rotation_started) {
            last_left_hand_rotation = left_hand_rotation;
            last_left_hand_translation = left_hand_translation;
        }

        glm::quat rotation_diff = (left_hand_rotation * glm::inverse(last_left_hand_rotation));
        glm::vec3 translation_diff = left_hand_translation - last_left_hand_translation;

        Node3D* node = static_cast<Node3D*>(selected_node);

        node->rotate_world(rotation_diff);
        node->translate(translation_diff);

        if (Input::get_trigger_value(HAND_RIGHT) > 0.5) {

            if (!scale_started) {
                last_hand_distance = hand_distance;
                scale_started = true;
            }

            float hand_distance_diff = hand_distance / last_hand_distance;
            node->scale(glm::vec3(hand_distance_diff));
            last_hand_distance = hand_distance;
        }
        else if (scale_started) {
            scale_started = false;
        }

        rotation_started = true;

        last_left_hand_rotation = left_hand_rotation;
        last_left_hand_translation = left_hand_translation;
    }

    // If rotation has stopped
    else if (rotation_started && !is_shift_left_pressed) {
        rotation_started = false;
    }
}

void GroupEditor::inspect_group(bool force)
{
    inspector->clear();

    auto& nodes = current_group->get_children();

    for (auto node : nodes) {
        inspect_node(node);
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    inspector_dirty = false;
    inspector_transform_dirty = !inspector->get_visibility() || force;

    if (force) {
        inspector->set_visibility(true);
    }
}

void GroupEditor::inspect_node(Node* node, const std::string& texture_path)
{
    inspector->same_line();

    // add unique identifier for signals
    std::string node_name = node->get_name();

    std::string signal = node_name + "_label";
    uint32_t flags = ui::TEXT_EVENTS | (node == selected_node ? ui::SELECTED : 0);
    inspector->label(signal, node_name, flags);

    Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
        select_node(n, false);
    });

    // Delete from group button
    signal = node_name + std::to_string(node_signal_uid++) + "_ungroup";
    inspector->button(signal, "data/textures/cross.png", ui::CONFIRM_BUTTON);

    Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
        ungroup_node(n);
    });

    inspector->end_line();
}
