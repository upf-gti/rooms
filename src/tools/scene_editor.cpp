#include "scene_editor.h"

#include "framework/input.h"
#include "framework/parsers/parse_scene.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/character_3d.h"
#include "framework/nodes/player_node.h"
#include "framework/nodes/spot_light_3d.h"
#include "framework/nodes/omni_light_3d.h"
#include "framework/nodes/directional_light_3d.h"
#include "framework/nodes/group_3d.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/math/intersections.h"
#include "framework/ui/io.h"
#include "framework/ui/inspector.h"
#include "framework/ui/keyboard.h"
#include "framework/ui/context_menu.h"
#include "framework/camera/camera.h"
#include "framework/resources/room.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "graphics/managers/sculpt_manager.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <filesystem>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtc/constants.hpp>

#define MAX_UNDO_STEPS 64

uint64_t SceneEditor::node_signal_uid = 0;

Color SceneEditor::COLOR_HIGHLIGHT_NODE = Color(0.929f, 0.778f, 0.6f, 1.0f);
Color SceneEditor::COLOR_HIGHLIGHT_GROUP = Color(0.187f, 0.089f, 0.858f, 1.0f);
Color SceneEditor::COLOR_HIGHLIGHT_LIGHT = Color(1.0f, 0.136f, 0.0f, 1.0f);
Color SceneEditor::COLOR_HIGHLIGHT_CHARACTER = Color(0.154f, 0.85836f, 0.235f, 1.0f);

void SceneEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    main_scene = Engine::instance->get_main_scene();

    current_room = new Room(main_scene);
    current_room->ref();

    gizmo = static_cast<RoomsEngine*>(RoomsEngine::instance)->get_gizmo();

    init_ui();

#ifndef DISABLE_RAYMARCHER
    SculptNode* default_sculpt = new SculptNode();
    default_sculpt->set_name("default_sculpt");
    default_sculpt->initialize();
    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    default_sculpt->set_position({0.0f, 1.0f, 0.0f});
    //static_cast<RoomsEngine*>(RoomsEngine::instance)->set_current_sculpt(default_sculpt);
    main_scene->add_node(default_sculpt);
#endif

   /* auto e = parse_mesh("data/meshes/torus/torus.obj");
    main_scene->add_node(e);

    e = parse_mesh("data/meshes/cube/cube.obj");
    main_scene->add_node(e);*/

    Node::bind(main_scene->get_name() + "@nodes_added", [&](const std::string& sg, void* data) {
        set_inspector_dirty();
    });

    Node::bind("@node_deleted", [&](const std::string& sg, void* data) {
        Node* node = reinterpret_cast<Node*>(data);
        if (node == selected_node) {
            deselect();
        }
        set_inspector_dirty();
    });

    Node::bind("@on_gpu_results", [&](const std::string& sg, void* data) {

        // Do nothing if it's not the current editor..
        auto engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
        if (engine->get_current_editor() != this) {
            return;
        }

        hovered_node = nullptr;

        if (moving_node) {
            return;
        }

        sGPU_SculptResults* last_gpu_results = reinterpret_cast<sGPU_SculptResults*>(data);
        assert(last_gpu_results);
        sGPU_RayIntersectionData& intersection = last_gpu_results->ray_intersection;

        if (intersection.has_intersected == 0u) {
            return;
        }

        std::function<void(Node* node)> check_intersections = [&](Node* node) {
            SculptNode* sculpt_node = dynamic_cast<SculptNode*>(node);
            if (sculpt_node && sculpt_node->check_intersection(&intersection)) {
                // hover the group
                if (!current_group && sculpt_node->get_parent()) {
                    hovered_node = sculpt_node->get_parent();
                    // assert(dynamic_cast<Group3D*>(hovered_node));
                }
                else {
                    hovered_node = sculpt_node;
                }
                IO::set_xr_ray_distance(intersection.ray_t);
            }
            for (auto child : node->get_children()) {
                check_intersections(child);
            }
        };

        auto& nodes = current_group ? current_group->get_children() : main_scene->get_nodes();
        for (auto node : nodes) {
            check_intersections(node);
        }
    });
}

void SceneEditor::clean()
{
    BaseEditor::clean();

    // Clean nodes deleted still stored for undo/redo
    for (const auto& node_data : deleted_nodes) {
        delete node_data.second.ref;
    }

    deleted_nodes.clear();
}

void SceneEditor::update(float delta_time)
{
    BaseEditor::update(delta_time);

    // Update input actions
    {
        select_action_pressed = Input::was_mouse_released(GLFW_MOUSE_BUTTON_LEFT);

        if (renderer->get_openxr_available()) {
            eTriggerAction trigger_state = get_trigger_action(delta_time);
            select_action_pressed |= (trigger_state == TRIGGER_TAPPED);

            if (trigger_state == TRIGGER_HOLDED && prev_trigger_state == NO_TRIGGER_ACTION) {
                holded_node = hovered_node;
            }
            else if (trigger_state == NO_TRIGGER_ACTION && prev_trigger_state == TRIGGER_HOLDED) {
                holded_node = nullptr;
            }

            prev_trigger_state = trigger_state;
        }

        if (Input::was_key_pressed(GLFW_KEY_ESCAPE)) {
            deselect();
        }
    }

    if (exports_dirty) {
        get_export_files();
    }

    update_gizmo(delta_time);

    update_node_transform(delta_time, holded_node != nullptr);

    update_hovered_node();

    shortcuts.clear();
    shortcuts[shortcuts::TOGGLE_SCENE_INSPECTOR] = !is_shift_left_pressed;

    if(!selected_node || (hovered_node != selected_node)) {
        shortcuts[shortcuts::SELECT_NODE] = !is_shift_right_pressed;
    }

    if (hovered_node) {
        process_node_hovered();
    }
    else if (moving_node) {

        assert(selected_node);

        shortcuts[shortcuts::PLACE_NODE] = true;

        glm::vec3 pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);

        if (current_group || selected_node->get_parent()) {
            Transform parent_transform = get_group_global_transform(selected_node);
            pos -= parent_transform.get_position();
        }

        static_cast<Node3D*>(selected_node)->set_position(pos);

        if (Input::was_trigger_pressed(HAND_RIGHT)) {
            moving_node = false;
        }
    }

    if (Input::was_button_pressed(XR_BUTTON_Y) || Input::was_key_pressed(GLFW_KEY_I)) {
        set_inspector_dirty();
    }

    if (inspector_dirty) {

        if (selected_node && dynamic_cast<Light3D*>(selected_node)) {
            inspect_light();
        }
        if (selected_node && dynamic_cast<Character3D*>(selected_node)) {
            inspect_character();
        }
        else if (current_group) {
            inspect_group(true);
        }
        else {
            inspector_from_scene(true);
        }
    }

    // Manage Undo/Redo
    {
        bool can_undo = (!is_shift_left_pressed && Input::was_button_pressed(XR_BUTTON_X)) || (Input::is_key_pressed(GLFW_KEY_LEFT_CONTROL) && Input::was_key_pressed(GLFW_KEY_Z));
        bool can_redo = (is_shift_left_pressed && Input::was_button_pressed(XR_BUTTON_X)) || (Input::is_key_pressed(GLFW_KEY_LEFT_CONTROL) && Input::was_key_pressed(GLFW_KEY_Y));

        shortcuts[shortcuts::SCENE_UNDO] = !is_shift_left_pressed;
        shortcuts[shortcuts::SCENE_REDO] = is_shift_left_pressed;

        if (can_undo) {
            scene_undo();
        } else if (can_redo) {
            scene_redo();
        }
    }

    if (renderer->get_openxr_available()) {
        BaseEditor::update_shortcuts(shortcuts);
    }

    inspector->update(delta_time);
}

void SceneEditor::render()
{
    RoomsEngine::render_controllers();

    render_gizmo();

    BaseEditor::render();

    inspector->render();
}

void SceneEditor::render_gui()
{
    if (selected_node) {
        ImGui::Text("Selected Node: %s", selected_node->get_name().c_str());
    }

    if (hovered_node) {
        ImGui::Text("Hovered Node");
    }
}

void SceneEditor::on_enter(void* data)
{
    gizmo->set_operation(TRANSLATE);
    Node::emit_signal("combo_gizmo_modes@changed", (void*)"translate");
}

void SceneEditor::set_main_scene(Scene* new_scene)
{
    current_group = nullptr;
    current_character = nullptr;

    main_scene = new_scene;
}

void SceneEditor::update_hovered_node()
{
    prev_ray_dir = ray_direction;
    prev_ray_origin = ray_origin;

    // Send rays each frame to detect hovered sculpts and other
    Engine::instance->get_scene_ray(ray_origin, ray_direction);

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    rooms_renderer->get_sculpt_manager()->set_ray_to_test(ray_origin, ray_direction);
}

uint32_t SceneEditor::get_sculpt_context_flags(SculptNode* node)
{
    uint32_t flags = SCULPT_IN_SCENE_EDITOR;

    auto parent = node->get_parent();

    if (current_group) {
        if ((!parent || parent != (Node*)current_group)) {
            flags |= SCULPT_IS_OUT_OF_FOCUS;
        }
    }
    else if (current_character) {
        if ((!parent || parent != (Node*)current_character)) {
            flags |= SCULPT_IS_OUT_OF_FOCUS;
        }
    }
    else if (parent) {
        flags |= SCULPT_HOVER_CHECK_SIBLINGS;
    }

    return flags;
}

Color SceneEditor::get_node_highlight_color(Node* node)
{
    if (dynamic_cast<Light3D*>(node)) {
        return COLOR_HIGHLIGHT_LIGHT;
    }
    else if (dynamic_cast<Group3D*>(node)) {
        return COLOR_HIGHLIGHT_GROUP;
    }
    else if (dynamic_cast<Group3D*>(node)) {
        return COLOR_HIGHLIGHT_GROUP;
    }
    else if (dynamic_cast<Character3D*>(node)) {
        return COLOR_HIGHLIGHT_CHARACTER;
    }

    return COLOR_HIGHLIGHT_NODE;
}

void SceneEditor::process_node_hovered()
{
    const bool sculpt_hovered = !!dynamic_cast<SculptNode*>(hovered_node);
    const bool group_hovered = !sculpt_hovered && !!dynamic_cast<Group3D*>(hovered_node);
    const bool character_hovered = !sculpt_hovered && !group_hovered && !!dynamic_cast<Character3D*>(hovered_node);
    const bool a_pressed = Input::was_button_pressed(XR_BUTTON_A);
    const bool b_pressed = Input::was_button_pressed(XR_BUTTON_B);
    const bool should_open_context_menu = Input::was_button_pressed(XR_BUTTON_B) || Input::was_mouse_pressed(GLFW_MOUSE_BUTTON_RIGHT);

    if (group_hovered) {
        if (grouping_node) {
            bool can_group = (hovered_node != node_to_group);
            shortcuts[shortcuts::ADD_TO_GROUP] = can_group;
            if (a_pressed && can_group) {
                process_group();
            }
        }
        else {
            shortcuts[shortcuts::EDIT_GROUP] = true;
            if (b_pressed) {
                edit_group(static_cast<Group3D*>(hovered_node));
            }
            else if (select_action_pressed) {
                select_node(hovered_node, false);
            }
        }
    }
    else if (character_hovered) {
        if (should_open_context_menu) {
            glm::vec2 position = Input::get_mouse_position();
            glm::vec3 position_3d = glm::vec3(0.0f);

            if (renderer->get_openxr_available()) {
                position = { 0.0f, 0.0f };
                const sGPU_SculptResults& gpu_results = renderer->get_sculpt_manager()->loaded_results;
                position_3d = ray_origin + ray_direction * gpu_results.ray_intersection.ray_t;
            }

            new ui::ContextMenu(position, position_3d, {
                { "Animate", [&, n = hovered_node](const std::string& name, uint32_t index) { selected_node = n; RoomsEngine::switch_editor(ANIMATION_EDITOR, n); }},
                { "Delete", [&, n = hovered_node](const std::string& name, uint32_t index) { delete_node(n); }}
            });
        }
    }
    // In 2d, we have to select manually by click, so do not enter here!
    else if (grouping_node && renderer->get_openxr_available()) {
        bool can_group = (hovered_node != node_to_group);
        shortcuts[shortcuts::CREATE_GROUP] = can_group;
        if (a_pressed && can_group) {
            process_group();
        }
    }
    else if (is_shift_right_pressed) {
        shortcuts[shortcuts::CLONE_NODE] = true;
        shortcuts[shortcuts::UNGROUP] = !!current_group;
        shortcuts[shortcuts::GROUP_NODE] = !current_group;
        if (a_pressed) {
            clone_node(hovered_node, false);
        }
        else if (b_pressed) {
            if (current_group) {
                ungroup_node(hovered_node, true, false);
            }
            else {
                group_node(hovered_node);
            }
        }
    }
    else {
        shortcuts[shortcuts::OPEN_CONTEXT_MENU] = true;
        shortcuts[shortcuts::EDIT_SCULPT_NODE] = sculpt_hovered;
        if (a_pressed && sculpt_hovered) {
            select_node(hovered_node, false);
            RoomsEngine::switch_editor(SCULPT_EDITOR, static_cast<SculptNode*>(hovered_node));
        }
        else if (should_open_context_menu) {
            glm::vec2 position = Input::get_mouse_position();
            glm::vec3 position_3d = glm::vec3(0.0f);

            if (renderer->get_openxr_available()) {
                position = { 0.0f, 0.0f };
                const sGPU_SculptResults& gpu_results = renderer->get_sculpt_manager()->loaded_results;
                position_3d = ray_origin + ray_direction * gpu_results.ray_intersection.ray_t;
            }

            new ui::ContextMenu(position, position_3d, {
                { "Animate", [&, n = hovered_node](const std::string& name, uint32_t index) { selected_node = n; RoomsEngine::switch_editor(ANIMATION_EDITOR, n); }},
                { "Make Unique", [&, n = hovered_node](const std::string& name, uint32_t index) { make_unique(n); }},
                { "Delete", [&, n = hovered_node](const std::string& name, uint32_t index) { delete_node(n); }}
            });
        }
        else if (select_action_pressed) {
            select_node(hovered_node, false);
        }
    }
}

void SceneEditor::enter_room()
{
    RoomsEngine::switch_editor(PLAYER_EDITOR, current_room);
}

void SceneEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel = new ui::HContainer2D("scene_editor_root", { 48.0f, screen_size.y - 216.f }, ui::CREATE_3D);

    Node::bind("scene_editor_root@resize", (FuncUVec2)[&](const std::string& signal, glm::u32vec2 window_size) {
        main_panel->set_position({ 48.0f, window_size.y - 216.f });
    });

    ui::VContainer2D* vertical_container = new ui::VContainer2D("scene_vertical_container", { 0.0f, 0.0f });
    main_panel->add_child(vertical_container);

    // Add main rows
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    ui::HContainer2D* second_row = new ui::HContainer2D("row_1", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    first_row->add_child(new ui::TextureButton2D("deselect", { "data/textures/cross.png" }));

    // ** Undo/Redo scene **
    {
        ui::ItemGroup2D* g_scene_undo_redo = new ui::ItemGroup2D("g_scene_undo_redo");
        g_scene_undo_redo->add_child(new ui::TextureButton2D("scene_undo", { "data/textures/undo.png", ui::DISABLED }));
        g_scene_undo_redo->add_child(new ui::TextureButton2D("scene_redo", { "data/textures/redo.png", ui::DISABLED }));
        first_row->add_child(g_scene_undo_redo);
    }

    // ** Node actions **
    {
        ui::ButtonSubmenu2D* node_actions_submenu = new ui::ButtonSubmenu2D("node_actions", { "data/textures/cube.png", ui::DISABLED });
        ui::ItemGroup2D* g_grouping = new ui::ItemGroup2D("g_grouping");
        g_grouping->add_child(new ui::TextureButton2D("group", { "data/textures/group.png" }));
        g_grouping->add_child(new ui::TextureButton2D("ungroup", { "data/textures/ungroup.png" }));
        node_actions_submenu->add_child(g_grouping);
        // node_actions_submenu->add_child(new ui::TextureButton2D("duplicate", { "data/textures/clone.png" }));
        node_actions_submenu->add_child(new ui::TextureButton2D("clone", { "data/textures/clone_instance.png" }));
        first_row->add_child(node_actions_submenu);
    }

    // ** Posible scene nodes **
    {
        ui::ButtonSubmenu2D* add_node_submenu = new ui::ButtonSubmenu2D("add_node", { "data/textures/add.png" });

        add_node_submenu->add_child(new ui::TextureButton2D("sculpt", { "data/textures/sculpt.png" }));
        add_node_submenu->add_child(new ui::TextureButton2D("character", { "data/textures/character.png" }));

        // Lights
        {
            ui::ItemGroup2D* g_add_node = new ui::ItemGroup2D("g_light_types", ui::HIDDEN);
            g_add_node->add_child(new ui::TextureButton2D("omni", { "data/textures/light.png" }));
            g_add_node->add_child(new ui::TextureButton2D("spot", { "data/textures/spot.png"  }));
            g_add_node->add_child(new ui::TextureButton2D("directional", { "data/textures/sun.png"  }));
            add_node_submenu->add_child(g_add_node);
        }

        first_row->add_child(add_node_submenu);
    }

    // ** Display Settings **
    {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        ui::ButtonSubmenu2D* display_submenu = new ui::ButtonSubmenu2D("display", { "data/textures/display_settings.png", ui::HIDDEN });
        ui::ItemGroup2D* g_display = new ui::ItemGroup2D("g_display");
        g_display->add_child(new ui::TextureButton2D("use_grid", { "data/textures/grid.png", ui::ALLOW_TOGGLE | ui::SELECTED }));
        g_display->add_child(new ui::TextureButton2D("use_environment", { "data/textures/skybox.png", ui::ALLOW_TOGGLE | ui::SELECTED }));
        g_display->add_child(new ui::FloatSlider2D("IBL_intensity", { .path = "data/textures/ibl_intensity.png", .fvalue = rooms_renderer->get_ibl_intensity(), .flags = ui::USER_RANGE, .fvalue_max = 4.0f, .precision = 2 }));
        display_submenu->add_child(g_display);
        display_submenu->add_child(new ui::FloatSlider2D("exposure", { .path = "data/textures/exposure.png", .fvalue = rooms_renderer->get_exposure(), .flags = ui::USER_RANGE, .fvalue_max = 4.0f, .precision = 2 }));
        first_row->add_child(display_submenu);
    }

    // ** Gizmo modes **
    {
        ui::ComboButtons2D* combo_gizmo_modes = new ui::ComboButtons2D("combo_gizmo_modes");
        combo_gizmo_modes->add_child(new ui::TextureButton2D("no_gizmo", { "data/textures/no_gizmo.png" }));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("translate", { "data/textures/translation_gizmo.png", ui::SELECTED }));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("rotate", { "data/textures/rotation_gizmo.png" }));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("scale", { "data/textures/scale_gizmo.png" }));
        second_row->add_child(combo_gizmo_modes);
    }

    // ** Import/Export scene **
    {
        second_row->add_child(new ui::TextureButton2D("load", { "data/textures/load.png", ui::DISABLED }));
        second_row->add_child(new ui::TextureButton2D("save", { "data/textures/save.png", ui::DISABLED }));
    }

    // ** Player stuff **
    {
        second_row->add_child(new ui::TextureButton2D("enter_room", { "data/textures/play.png", ui::DISABLED }));
    }

    // Create inspection panel (Nodes, properties, etc)
    {
        inspector = new ui::Inspector({
            .name = "inspector_root",
            .title = "Scene Nodes",
            .position = {32.0f, 32.f},
            .close_fn = std::bind(&SceneEditor::on_close_inspector, this, std::placeholders::_1),
            .back_fn = std::bind(&SceneEditor::on_goback_inspector, this, std::placeholders::_1)
        });
        inspector->set_visibility(false);
    }

    main_panel->set_visibility(false);

    // Load controller UI labels
    if (renderer->get_openxr_available()) {
        // Thumbsticks
        // Buttons
        // Triggers

        glm::vec2 double_size = { 2.0f, 1.0f };

        // Left hand
        {
            left_hand_box = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            left_hand_box->add_childs({
                new ui::ImageLabel2D("Scene Panel", shortcuts::Y_BUTTON_PATH, shortcuts::TOGGLE_SCENE_INSPECTOR),
                new ui::ImageLabel2D("Redo", shortcuts::L_GRIP_X_BUTTON_PATH, shortcuts::SCENE_REDO, double_size),
                new ui::ImageLabel2D("Redo", shortcuts::L_GRIP_X_BUTTON_PATH, shortcuts::SCENE_REDO, double_size)
            });
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            right_hand_box->add_childs({
                new ui::ImageLabel2D("More Options", shortcuts::B_BUTTON_PATH, shortcuts::OPEN_CONTEXT_MENU),
                new ui::ImageLabel2D("Edit Sculpt", shortcuts::A_BUTTON_PATH, shortcuts::EDIT_SCULPT_NODE),
                new ui::ImageLabel2D("Edit Group", shortcuts::A_BUTTON_PATH, shortcuts::EDIT_GROUP),
                new ui::ImageLabel2D("Create Group", shortcuts::R_GRIP_B_BUTTON_PATH, shortcuts::ANIMATE_NODE, double_size),
                new ui::ImageLabel2D("Add to Group", shortcuts::R_GRIP_B_BUTTON_PATH, shortcuts::ANIMATE_NODE, double_size),
                new ui::ImageLabel2D("Clone Node", shortcuts::R_GRIP_A_BUTTON_PATH, shortcuts::CLONE_NODE, double_size),
                new ui::ImageLabel2D("Place Node", shortcuts::R_TRIGGER_PATH, shortcuts::PLACE_NODE),
                new ui::ImageLabel2D("Select Node", shortcuts::R_TRIGGER_PATH, shortcuts::SELECT_NODE),
                new ui::ImageLabel2D("Ungroup", shortcuts::R_GRIP_R_TRIGGER_PATH, shortcuts::UNGROUP, double_size),
                new ui::ImageLabel2D("Group Node", shortcuts::R_GRIP_R_TRIGGER_PATH, shortcuts::GROUP_NODE, double_size)
            });
        }
    }

    // Bind callbacks
    bind_events();
}

void SceneEditor::bind_events()
{
    // Undo/Redo
    {
        Node::bind("scene_undo", [&](const std::string& signal, void* button) { scene_undo(); });
        Node::bind("scene_redo", [&](const std::string& signal, void* button) { scene_redo(); });
    }

    Node::bind("sculpt", [&](const std::string& signal, void* button) {
        SculptNode* new_sculpt = new SculptNode();
        new_sculpt->initialize();
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        rooms_renderer->toogle_frame_debug();

        static_cast<RoomsEngine*>(RoomsEngine::instance)->set_current_sculpt(new_sculpt);

        select_node(new_sculpt);
        add_node(new_sculpt);
        set_inspector_dirty();

        Node::emit_signal("@on_sculpt_added", (void*)nullptr);
    });

    Node::bind("character", [&](const std::string& signal, void* button) {
        Character3D* new_character = new Character3D();
        new_character->initialize();
        select_node(new_character);
        add_node(new_character);
        set_inspector_dirty();
    });

    // Environment / Scene Lights
    {
        Node::bind("omni", [&](const std::string& signal, void* button) { create_light_node(LIGHT_OMNI); });
        Node::bind("spot", [&](const std::string& signal, void* button) { create_light_node(LIGHT_SPOT); });
        Node::bind("directional", [&](const std::string& signal, void* button) { create_light_node(LIGHT_DIRECTIONAL); });

        Node::bind("exposure", (FuncFloat)[&](const std::string& signal, float value) {
            RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
            rooms_renderer->set_exposure(value);
        });

        Node::bind("IBL_intensity", (FuncFloat)[&](const std::string& signal, float value) {
            RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
            rooms_renderer->set_ibl_intensity(value);
        });

        Node::bind("use_environment", [&](const std::string& signal, void* button) {
            RoomsEngine::toggle_use_environment_map();
        });

        Node::bind("use_grid", [&](const std::string& signal, void* button) {
            RoomsEngine::toggle_use_grid();
        });
    }

    Node::bind("deselect", [&](const std::string& signal, void* button) { deselect(); });
    Node::bind("group", [&](const std::string& signal, void* button) { group_node(selected_node); });
    Node::bind("ungroup", [&](const std::string& signal, void* button) { ungroup_node(selected_node, true, false); });
    // Node::bind("duplicate", [&](const std::string& signal, void* button) { clone_node(selected_node, true); });
    Node::bind("clone", [&](const std::string& signal, void* button) { clone_node(selected_node, false); });

    // Export / Import (.room) / Player
    {
        auto callback = [&](const std::string& output) {
            main_scene->serialize("data/exports/" + output + ".room");
            main_scene->set_name(output);
            exports_dirty = true;
        };

        Node::bind("save", [&, fn = callback](const std::string& signal, void* button) { ui::Keyboard::request(fn, main_scene->get_name()); });
        Node::bind("load", [&](const std::string& signal, void* button) { inspect_exports(true); });
        Node::bind("enter_room", [&](const std::string& signal, void* button) { enter_room(); });
    }

    // Reactive buttons on finish tutorial
    {
        Node::bind("@on_tutorial_ended", [&](const std::string& signal, void* button) {

            main_panel->set_visibility(true);

            std::vector<std::string> to_enable = { "load", "save", "enter_room", "node_actions", "scene_undo", "scene_redo" };
            for (const std::string& w : to_enable) {
                Node2D::get_widget_from_name<ui::ButtonSubmenu2D*>(w)->set_disabled(false);
            }

            std::vector<std::string> to_show = { "display", "g_light_types" };
            for (const std::string& w : to_show) {
                Node2D::get_widget_from_name(w)->set_visibility(true);
            }
        });

        Node::bind("@on_tutorial_step", (FuncInt)[&](const std::string& signal, int step) {

            switch (step)
            {
            case TUTORIAL_ADD_NODE:
                main_panel->set_visibility(true);
                break;
            /*case TUTORIAL_STAMP_SMEAR:
                break;*/
            }
        });
    }
}

bool SceneEditor::on_goback_inspector(ui::Inspector* scope)
{
    if (current_group) {
        current_group = nullptr;
        Node2D::get_widget_from_name<ui::TextureButton2D*>("group")->set_disabled(false);
    }

    if (current_character) {
        current_character = nullptr;
    }

    deselect();

    set_inspector_dirty();

    return true;
}

bool SceneEditor::on_close_inspector(ui::Inspector* scope)
{
    if (current_group) {
        current_group = nullptr;
        Node2D::get_widget_from_name<ui::TextureButton2D*>("group")->set_disabled(false);
    }

    if (current_character) {
        current_character = nullptr;
    }

    return true;
}

void SceneEditor::select_node(Node* node, bool place)
{
    // Avoid input issues with interacting with gizmo
    if (IO::get_want_capture_input()) {
        return;
    }

    // Select group target node in 2d only!
    if (grouping_node && !renderer->get_openxr_available()) {
        process_group(node);
        return;
    }

    selected_node = node;

    // Update inspector selection
    Node::emit_signal(node->get_name() + "_label@selected", (void*)nullptr);

    // To allow the user to move the node at the beginning
    moving_node = place && is_gizmo_usable() && renderer->get_openxr_available();
}

void SceneEditor::add_node(Node* node, Node* parent, int idx)
{
    Node3D* node_3d = static_cast<Node3D*>(node);

    // Check for its own group first
    if (parent) {
        Group3D* group = static_cast<Group3D*>(parent);
        node_3d->set_transform(Transform::combine(node_3d->get_transform(), group->get_transform()));
        group->add_node(node_3d);
    }
    else if (current_group) {
        node_to_group = node;
        node_3d->set_transform(Transform::combine(node_3d->get_transform(), current_group->get_transform()));
        process_group(current_group, false);
    }
    else {
        main_scene->add_node(node, idx);
    }
}

void SceneEditor::deselect()
{
    // hack by now: Do nothing if it's not the current editor..
    auto engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
    if (engine->get_current_editor() != this) {
        return;
    }

    selected_node = nullptr;

    moving_node = false;

    ui::Text2D::select(nullptr);
}

Node* SceneEditor::clone_node(Node* node, bool copy, bool push_undo)
{
    /*if (!selected_node) {
        return nullptr;
    }*/

    RoomsEngine* engine = static_cast<RoomsEngine*>(RoomsEngine::instance);

    Node* new_node = NodeRegistry::get_instance()->create_node(node->get_node_type());
    node->clone(new_node, copy);

    SculptNode* sculpt_node = dynamic_cast<SculptNode*>(new_node);

    if (sculpt_node) {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        rooms_renderer->toogle_frame_debug();
        engine->set_current_sculpt(sculpt_node);
    }

    add_node(new_node);

    select_node(new_node);

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_CLONE, node, new_node, copy });
    }

    set_inspector_dirty();

    return new_node;
}

void SceneEditor::make_unique(Node* node)
{
    // TODO: Maybe this should be a overridable mwthod in Node since each one will make it differntly
    // For now support only sculpts

    SculptNode* sculpt_node = dynamic_cast<SculptNode*>(node);

    if (!sculpt_node) {
        return;
    }

    sculpt_node->make_unique();
}

void SceneEditor::group_node(Node* node)
{
    if (!node) {
        return;
    }

    /*
    *   The idea is now to hover and accept another node:
    *   - If it does not have a group, create and group both nodes into it
    *   - If it already has group, put the first node in the second one's group
    */

    grouping_node = true;
    node_to_group = node;
}

void SceneEditor::delete_node(Node* node, bool push_undo)
{
    if (selected_node == node) {
        deselect();
    }

    uintptr_t idx = reinterpret_cast<uintptr_t>(node);

    Node* parent = node->get_parent();

    if (parent) {
        parent->remove_child(node);
    }
    else {
        main_scene->remove_node(node);
    }

    // TODO: Store correct index
    deleted_nodes[idx] = { -1, node };

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_DELETE, node, parent }, false);
    }

    set_inspector_dirty();
}

void SceneEditor::recover_node(Node* node, Node* parent, bool push_redo)
{
    uintptr_t idx = reinterpret_cast<uintptr_t>(node);

    if (!deleted_nodes.contains(idx)) {
        return;
    }

    const sDeletedNode& node_data = deleted_nodes[idx];
    Node* recovered_node = node_data.ref;
    add_node(recovered_node, parent, node_data.index);
    deleted_nodes.erase(idx);

    if (push_redo) {
        push_redo_action({ sActionData::ACTION_DELETE, recovered_node });
    }

    set_inspector_dirty();
}

void SceneEditor::ungroup_node(Node* node, bool push_undo, bool push_redo)
{
    assert(node);

    if (dynamic_cast<Group3D*>(node)) {
        ungroup_all(node);
        return;
    }

    if (!node->get_parent()) {
        return;
    }

    Node3D* g_node = static_cast<Node3D*>(node);
    Group3D* group = g_node->get_parent<Group3D*>();

    glm::mat4x4 world_space_model = g_node->get_global_model();
    group->remove_child(g_node);
    g_node->set_transform(Transform::mat4_to_transform(world_space_model));

    main_scene->add_node(node);

    sActionData data = { sActionData::ACTION_GROUP, g_node, group };

    // Remove group since it has been created for this node..
    if (push_redo && group->get_children().size() == 1u) {
        Node* child = group->get_children().front();
        data.param_2 = child;
        ungroup_node(child, false, false);
    }
    else if (group->get_children().empty()) {
        delete_node(group, false);
    }

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_UNGROUP, g_node, group });
    }

    if (push_redo) {
        push_redo_action(data);
    }

    deselect();

    set_inspector_dirty();
}

// This cannot be undone by now
void SceneEditor::ungroup_all(Node* node)
{
    Group3D* g_node = static_cast<Group3D*>(node);

    for (auto child : g_node->get_children()) {
        ungroup_node(child, false, false);
    }
}

void SceneEditor::process_group(Node* node, bool push_undo)
{
    Node3D* hovered_3d = static_cast<Node3D*>(node ? node : hovered_node);
    Node3D* to_group_3d = static_cast<Node3D*>(node_to_group);

    // same node!
    if (hovered_3d == to_group_3d) {
        return;
    }

    grouping_node = false;

    // Check if current hover has group... (parent)
    Group3D* group = dynamic_cast<Group3D*>(hovered_3d);

    if (group) {
        // Add first node to the same group as the current hover
        main_scene->remove_node(to_group_3d);
        group->add_node(to_group_3d);
        select_node(group, false);
    }
    else {
        // Remove nodes from main scene
        main_scene->remove_node(to_group_3d);
        main_scene->remove_node(hovered_3d);

        // Create new group and add the nodes
        Group3D* new_group = new Group3D();
        new_group->add_nodes({ to_group_3d, hovered_3d });
        main_scene->add_node(new_group);

        select_node(new_group, false);
    }

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_GROUP, to_group_3d });
    }

    set_inspector_dirty();
}

void SceneEditor::edit_group(Group3D* group)
{
    deselect();

    current_group = group;

    set_inspector_dirty();

    Node2D::get_widget_from_name<ui::TextureButton2D*>("group")->set_disabled(true);
}

const Transform& SceneEditor::get_group_global_transform(Node* node)
{
    if (current_group) {
        return current_group->get_transform();
    }

    else if (node->get_parent()) {
        return node->get_parent<Node3D*>()->get_transform();
    }

    assert(0);

    return Transform();
}

void SceneEditor::create_light_node(uint8_t type)
{
    Light3D* new_light = nullptr;

    switch (type)
    {
    case LIGHT_OMNI:
        new_light = new OmniLight3D();
        new_light->set_position({ 1.0f, 1.f, 0.0f });
        break;
    case LIGHT_SPOT:
        new_light = new SpotLight3D();
        new_light->set_position({ 0.0f, 1.f, 0.0f });
        new_light->rotate(glm::radians(-90.f), { 1.f, 0.0f, 0.f });
        break;
        case LIGHT_DIRECTIONAL:
        new_light = new DirectionalLight3D();
        new_light->rotate(glm::radians(-90.f), { 1.f, 0.0f, 0.f });
        break;
    default:
        spdlog::error("Unsupported light type: {}", type);
        assert(0);
        break;
    }

    new_light->set_color({ 1.0f, 1.0f, 1.0f });
    new_light->set_intensity(1.0f);
    new_light->set_range(5.0f);

    add_node(new_light);

    select_node(new_light);

    set_inspector_dirty();
}

bool SceneEditor::is_gizmo_usable()
{
    bool r = !!selected_node;

    if (r) {
        r &= !!dynamic_cast<Node3D*>(selected_node);
    }

    return r;
}

void SceneEditor::update_gizmo(float delta_time)
{
    if (!is_gizmo_usable()) {
        return;
    }

    // Gizmo should update in XR mode only 
    if (!renderer->get_openxr_available()) {
        return;
    }

    Node3D* node = static_cast<Node3D*>(selected_node);
    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);

    Transform t = node->get_transform();
    Transform parent_transform;

    if (current_group || node->get_parent()) {
        parent_transform = get_group_global_transform(node);
        t = Transform::combine(parent_transform, t);
    }

    if (gizmo->update(t, right_controller_pos, delta_time)) {

        if (!action_in_progress) {
            push_undo_action({ sActionData::ACTION_TRANSFORM, node, node->get_transform() });
            action_in_progress = true;
        }

        if (current_group || node->get_parent()) {
            t = Transform::combine(Transform::inverse(parent_transform), t);
        }

        node->set_transform(t);
    }
    else if (action_in_progress) {
        action_in_progress = false;
    }
}

void SceneEditor::render_gizmo()
{
    if (!is_gizmo_usable()) {
        return;
    }

    Node3D* node = static_cast<Node3D*>(selected_node);

    Transform t = node->get_transform();
    Transform parent_transform;

    if (current_group || node->get_parent()) {
        parent_transform = get_group_global_transform(node);
        t = Transform::combine(parent_transform, t);
    }

    gizmo->set_transform(t);

    bool transform_dirty = gizmo->render();

    // This is only for 2D since Gizmo.render will only return true if
    // Gizmo2D is used!
    if (renderer->get_openxr_available()) {
        return;
    }

    if (transform_dirty) {

        if (!action_in_progress) {
            push_undo_action({ sActionData::ACTION_TRANSFORM, node, node->get_transform() });
            action_in_progress = true;
        }

        Transform new_transform = gizmo->get_transform();

        if (current_group || node->get_parent()) {
            new_transform = Transform::combine(Transform::inverse(parent_transform), new_transform);
        }

        node->set_transform(new_transform);
    }
    else if (action_in_progress && !Input::is_mouse_pressed(GLFW_MOUSE_BUTTON_LEFT)) {
        action_in_progress = false;
    }
}

void SceneEditor::update_node_transform(const float delta_time, const bool rotate_selected_node)
{
    // Do not rotate sculpt if shift -> we might be rotating the edit
    if (rotate_selected_node && !is_shift_left_pressed) {

        if (!selected_node) {
            if (!hovered_node) {
                return;
            }
            selected_node = hovered_node;
        }

        Node3D* node_3d = static_cast<Node3D*>(selected_node);

        glm::quat right_hand_rotation = Input::get_controller_rotation(HAND_RIGHT, POSE_AIM);
        glm::vec3 right_hand_translation = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
        glm::vec3 left_hand_translation = Input::get_controller_position(HAND_LEFT, POSE_AIM);
        float hand_distance = glm::length2(right_hand_translation - left_hand_translation);

        if (!rotation_started) {
            RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
            const float ray_t = rooms_renderer->get_sculpt_manager()->loaded_results.ray_intersection.ray_t;
            const glm::vec3 sculpt_intersection_pos = prev_ray_origin + prev_ray_dir * ray_t;

            last_right_hand_rotation = right_hand_rotation;
            last_right_hand_translation = right_hand_translation;
            hand_sculpt_distance = glm::length(node_3d->get_translation() - right_hand_translation);
            push_undo_action({ sActionData::ACTION_TRANSFORM, node_3d, node_3d->get_transform() });
        }

        const glm::vec2 thumbstick_values = Input::get_thumbstick_value(HAND_RIGHT);

        if (glm::abs(thumbstick_values.y) >= THUMBSTICK_DEADZONE) {
            const float new_distance_delta = (thumbstick_values.y + glm::sign(thumbstick_values.y) * THUMBSTICK_DEADZONE) * delta_time;
            hand_sculpt_distance = glm::clamp(hand_sculpt_distance + new_distance_delta, 0.0f, 5.0f);
        }

        glm::quat hand_rotation_diff = (right_hand_rotation * glm::inverse(last_right_hand_rotation));

        node_3d->set_position(hand_sculpt_distance * ray_direction + ray_origin);
        node_3d->get_transform().rotate_world(hand_rotation_diff);

        /*if (Input::get_trigger_value(HAND_RIGHT) > 0.5) {

            if (!scale_started) {
                last_hand_distance = hand_distance;
                scale_started = true;
            }

            float hand_distance_diff = hand_distance / last_hand_distance;
            node_3d->scale(glm::vec3(hand_distance_diff));
            last_hand_distance = hand_distance;
        }
        else if (scale_started) {
            scale_started = false;
        }*/

        rotation_started = true;

        last_right_hand_rotation = right_hand_rotation;
        last_right_hand_translation = right_hand_translation;
    }

    // If rotation has stopped
    else if (rotation_started && !is_shift_left_pressed) {
        rotation_started = false;
    }
}

void SceneEditor::inspector_from_scene(bool force)
{
    uint8_t flags = ui::INSPECTOR_FLAG_CLOSE_BUTTON;

    if (!inspector->get_visibility() || force) {
        flags |= ui::INSPECTOR_FLAG_FORCE_3D_POSITION;
    }

    inspector->clear(flags, "Scene Nodes");

    auto& nodes = main_scene->get_nodes();

    for (auto node : nodes) {

        if (dynamic_cast<Light3D*>(node)) {
            inspect_node(node, NODE_LIGHT);
        }
        else if (dynamic_cast<SculptNode*>(node)) {
            inspect_node(node, NODE_SCULPT);
        }
        else if (dynamic_cast<Group3D*>(node)) {
            inspect_node(node, NODE_GROUP);
        }
        else if (dynamic_cast<Character3D*>(node)) {
            inspect_node(node, NODE_CHARACTER);
        }
        else if (dynamic_cast<PlayerNode*>(node)) {
            continue;
        }
        else {
            inspect_node(node);
        }
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    inspector_dirty = false;

    if (force) {
        inspector->set_visibility(true);
        Node::emit_signal("@on_inspector_opened", (void*)nullptr);
    }
}

void SceneEditor::inspect_node(Node* node, uint32_t flags, const std::string& texture_path)
{
    inspector->same_line();

    // add unique identifier for signals
    std::string node_name = node->get_name();

    if ((flags & NODE_ICON) && texture_path.size()) {
        inspector->icon(texture_path);
    }
    else if (flags & NODE_SUBMENU_ICON) {
        inspector->icon("data/textures/cursors/dot_small.png");
    }

    //if (flags & NODE_VISIBILITY) {
    //    std::string signal = node_name + std::to_string(node_signal_uid++) + "_visibility";
    //    inspector->button(signal, "data/textures/visibility.png", ui::ALLOW_TOGGLE);

    //    Node::bind(signal, [n = node](const std::string& sg, void* data) {
    //        // Implement visibility for Node3D
    //        // ...
    //        });
    //}

    if (flags & NODE_EDIT) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_edit";
        inspector->button(signal, "data/textures/edit.png", 0u, "Edit");

        Node::bind(signal, [&, n = node, flags = flags](const std::string& sg, void* data) {

            select_node(n, false);

            // Set as current sculpt and go to sculpt editor
            if (dynamic_cast<SculptNode*>(n)) {
                Node::emit_signal("@on_sculpt_edited", (void*)nullptr);
                RoomsEngine::switch_editor(SCULPT_EDITOR, static_cast<SculptNode*>(n));
            }
            else if (dynamic_cast<Group3D*>(n)) {
                edit_group(static_cast<Group3D*>(n));
            }
            else if (dynamic_cast<Character3D*>(n)) {
                edit_character(static_cast<Character3D*>(n));
            }
        });
    }

    if (flags & NODE_ANIMATE) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_animate";
        inspector->button(signal, "data/textures/animate.png", 0u, "Animate");

        Node::bind(signal, [&, n = node, flags = flags](const std::string& sg, void* data) {

            selected_node = n;

            // TODO CHECK: This corrupts the pointer to the node "n"
            // select_node(n, false);

            RoomsEngine::switch_editor(ANIMATION_EDITOR, n);
        });
    }

    if (flags & NODE_NAME) {
        std::string signal = node_name + "_label";
        uint32_t flags = ui::TEXT_EVENTS | (node == selected_node ? ui::SELECTED : 0);
        inspector->label(signal, node_name, flags, SceneEditor::get_node_highlight_color(node));

        // Request keyboard and use the result to set the new node name. Not the nicest code, but anyway..
        {
            Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
                select_node(n, false);
            });

            auto callback = [&, n = node](const std::string& output) {
                n->set_name(output);
                set_inspector_dirty();
            };

            Node::bind(signal + "@long_click", [fn = callback, str = node_name](const std::string& sg, void* data) {
                ui::Keyboard::request(fn, str, 24u);
            });
        }
    }

    // Remove button
    if (flags & NODE_DELETE) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_remove";
        inspector->button(signal, "data/textures/delete.png", ui::CONFIRM_BUTTON, "Delete");

        Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
            delete_node(n);
        });
    }

    inspector->end_line();

    // Show children
    if (flags & NODE_CHILDREN) {

        auto& nodes = node->get_children();

        inspector->set_indentation(1u);

        for (auto node : nodes) {

            if (dynamic_cast<Light3D*>(node)) {
                inspect_node(node, NODE_LIGHT | NODE_SUBMENU_ICON);
            }
            else if (dynamic_cast<SculptNode*>(node)) {
                inspect_node(node, NODE_SCULPT | NODE_SUBMENU_ICON);
            }
            else {
                inspect_node(node, NODE_SUBMENU_ICON);
            }
        }

        inspector->set_indentation(0u);
    }
}

void SceneEditor::inspect_group(bool force)
{
    assert(current_group);

    uint8_t flags = ui::INSPECTOR_FLAG_BACK_BUTTON | ui::INSPECTOR_FLAG_CLOSE_BUTTON;

    if (!inspector->get_visibility()) {
        flags |= ui::INSPECTOR_FLAG_FORCE_3D_POSITION;
    }

    inspector->clear(flags, "Group Nodes");

    auto& nodes = current_group->get_children();

    for (auto node : nodes) {

        if (dynamic_cast<Light3D*>(node)) {
            inspect_node(node, NODE_LIGHT);
        }
        else if (dynamic_cast<SculptNode*>(node)) {
            inspect_node(node, NODE_SCULPT);
        }
        else {
            inspect_node(node);
        }
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    inspector_dirty = false;

    if (force) {
        inspector->set_visibility(true);
    }
}


void SceneEditor::inspect_character(bool force)
{
    if (!dynamic_cast<Character3D*>(selected_node)) {
        assert(0);
        return;
    }

    uint8_t flags = ui::INSPECTOR_FLAG_BACK_BUTTON | ui::INSPECTOR_FLAG_CLOSE_BUTTON;

    if (!inspector->get_visibility()) {
        flags |= ui::INSPECTOR_FLAG_FORCE_3D_POSITION;
        inspector->set_visibility(true);
    }

    inspector->clear(flags, "Character");

    auto character = static_cast<Character3D*>(selected_node);
    auto sculpt_nodes = character->get_children();

    for (uint32_t i = 0; i < sculpt_nodes.size(); ++i) {

        auto node = sculpt_nodes[i];

        inspector->same_line();

        {
            std::string signal = node->get_name() + std::to_string(node_signal_uid++) + "_edit_character_set";
            inspector->button(signal, "data/textures/edit.png", 0u, "Edit");

            Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
                RoomsEngine::switch_editor(SCULPT_EDITOR, static_cast<SculptNode*>(n));
                });
        }

        {
            std::string signal = node->get_name() + std::to_string(node_signal_uid++) + "_label_character_set";
            inspector->label(signal, node->get_name(), 0u, SceneEditor::COLOR_HIGHLIGHT_CHARACTER);
        }

        inspector->end_line();
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    inspector_dirty = false;

    if (force) {
        inspector->set_visibility(true);
    }
}

void SceneEditor::inspect_light(bool force)
{
    uint8_t flags = ui::INSPECTOR_FLAG_BACK_BUTTON | ui::INSPECTOR_FLAG_CLOSE_BUTTON;

    if (!inspector->get_visibility()) {
        flags |= ui::INSPECTOR_FLAG_FORCE_3D_POSITION;
    }

    Light3D* light = static_cast<Light3D*>(selected_node);
    std::string node_name = light->get_name();

    inspector->clear(flags, node_name);

    inspector->same_line();
    inspector->icon("data/textures/light.png");
    inspector->label("empty", node_name);
    inspector->end_line();

    // Color
    {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_picker";
        inspector->color_picker(signal, Color(light->get_color(), 1.0f));
        Node::bind(signal, [l = light](const std::string& sg, const Color& color) {
            l->set_color(color);
        });
    }

    // Intensity
    {
        inspector->same_line();
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
        inspector->fslider(signal, light->get_intensity(), nullptr, 0.0f, 10.0f, 2);
        inspector->label("empty", "Intensity");
        inspector->end_line();
        Node::bind(signal, (FuncFloat)[l = light](const std::string& sg, float value) {
            l->set_intensity(value);
        });
    }

    // Range
    {
        inspector->same_line();
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_range_slider";
        inspector->fslider(signal, light->get_intensity(), nullptr, 0.0f, 5.0f, 2);
        inspector->label("empty", "Range");
        inspector->end_line();
        Node::bind(signal, (FuncFloat)[l = light](const std::string& sg, float value) {
            l->set_range(value);
        });
    }

    inspector->end_line();

    inspector_dirty = false;

    if (force) {
        inspector->set_visibility(true);
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);
}

void SceneEditor::inspect_exports(bool force)
{
    uint8_t flags = ui::INSPECTOR_FLAG_BACK_BUTTON | ui::INSPECTOR_FLAG_CLOSE_BUTTON;

    if (!inspector->get_visibility()) {
        flags |= ui::INSPECTOR_FLAG_FORCE_3D_POSITION;
    }
    else {
        // In case of files being already displayed, a second click will update the files..
        get_export_files();
    }

    inspector->clear(flags, "Available Rooms");

    for (const std::string& name : exported_scenes) {

        std::string full_name = "data/exports/" + name;

        inspector->same_line();

        std::string signal = name + std::to_string(node_signal_uid++) + "_load";
        inspector->button(signal, "data/textures/load.png", 0u, "New Scene");
        Node::bind(signal, [&, str = full_name](const std::string& sg, void* data) {
            deselect();
            static_cast<RoomsEngine*>(RoomsEngine::instance)->set_main_scene(str);
            set_inspector_dirty();
        });

        signal = name + std::to_string(node_signal_uid++) + "_add";
        inspector->button(signal, "data/textures/add.png", 0u, "Add");
        Node::bind(signal, [&, str = full_name](const std::string& sg, void* data) {
            static_cast<RoomsEngine*>(RoomsEngine::instance)->add_to_main_scene(str);
        });

        inspector->label("empty", name);
        inspector->end_line();
    }

    if (force) {
        inspector->set_visibility(true);
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    inspector->clear_scroll();
}

void SceneEditor::get_export_files()
{
    exported_scenes.clear();

    std::string path = "data/exports/";

    if (!std::filesystem::is_directory(path)) {
        std::filesystem::create_directory(path);
    }

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string file_name = entry.path().string();
        std::string scene_name = file_name.substr(13);
        exported_scenes.push_back(scene_name);
    }

    exports_dirty = false;
}

void SceneEditor::push_undo_action(const sActionData& step, bool clear_redo)
{
    if (undo_list.size() >= MAX_UNDO_STEPS) {
        undo_list.erase(undo_list.begin());
    }

    undo_list.push_back(step);

    if (clear_redo) {
        redo_list.clear();
    }
}

void SceneEditor::push_redo_action(const sActionData& step)
{
    redo_list.push_back(step);
}

bool SceneEditor::scene_undo()
{
    if (undo_list.empty()) {
        return false;
    }

    sActionData step = undo_list.back();
    undo_list.pop_back();

    if (!step.ref_node) {
        return false;
    }

    switch (step.type)
    {
    case sActionData::ACTION_TRANSFORM:
    {
        Node3D* node_3d = static_cast<Node3D*>(step.ref_node);
        push_redo_action({ sActionData::ACTION_TRANSFORM, node_3d, node_3d->get_transform() });
        node_3d->set_transform(std::get<Transform>(step.param_1));
        break;
    }
    case sActionData::ACTION_DELETE:
    {
        Node* parent = std::get<Node*>(step.param_1);
        recover_node(step.ref_node, parent);
        break;
    }
    case sActionData::ACTION_GROUP:
    {
        ungroup_node(step.ref_node, false);
        break;
    }
    case sActionData::ACTION_CLONE:
    {
        Node* node_to_delete = std::get<Node*>(step.param_1);
        push_redo_action({ sActionData::ACTION_CLONE, step.ref_node, node_to_delete, std::get<bool>(step.param_2) });
        delete_node(node_to_delete, false);
        break;
    }
    case sActionData::ACTION_UNGROUP:
    {
        Node3D* group = std::get<Node3D*>(step.param_1);
        recover_node(group, nullptr, false);
        push_redo_action({ sActionData::ACTION_UNGROUP, step.ref_node, group });
        node_to_group = step.ref_node;
        process_group(group, false);
        break;
    }
    default:
        assert(0);
        break;
    }

    return true;
}

bool SceneEditor::scene_redo()
{
    if (redo_list.empty()) {
        return false;
    }

    sActionData step = redo_list.back();
    redo_list.pop_back();

    if (!step.ref_node) {
        return false;
    }

    switch (step.type)
    {
    case sActionData::ACTION_TRANSFORM:
    {
        Node3D* node_3d = static_cast<Node3D*>(step.ref_node);
        push_undo_action({ sActionData::ACTION_TRANSFORM, node_3d, node_3d->get_transform() }, false);
        node_3d->set_transform(std::get<Transform>(step.param_1));
        break;
    }
    case sActionData::ACTION_DELETE:
    {
        delete_node(step.ref_node); // delete again node
        break;
    }
    case sActionData::ACTION_GROUP:
    {
        Node3D* group = std::get<Node3D*>(step.param_1);
        recover_node(group, nullptr, false);
        push_undo_action({ sActionData::ACTION_GROUP, step.ref_node, group }, false);
        node_to_group = step.ref_node;
        process_group(group, false);
        Node* sec = std::get<Node*>(step.param_2);
        if (sec) {
            node_to_group = sec;
            process_group(group, false);
        }
        break;
    }
    case sActionData::ACTION_CLONE:
    {
        Node* node_to_recover = std::get<Node*>(step.param_1);
        bool copy = std::get<bool>(step.param_2);
        recover_node(node_to_recover, nullptr, false);
        push_undo_action({ sActionData::ACTION_CLONE, step.ref_node, node_to_recover, copy }, false);
        break;
    }
    case sActionData::ACTION_UNGROUP:
    {
        Node3D* group = std::get<Node3D*>(step.param_1);
        push_undo_action({ sActionData::ACTION_UNGROUP, step.ref_node, group }, false);
        ungroup_node(step.ref_node, false, false);
        break;
    }
    default:
        assert(0);
        break;
    }

    return true;
}

void SceneEditor::edit_character(Character3D* character)
{
    if (!character) {
        assert(0);
        return;
    }

    current_character = character;

    set_inspector_dirty();
}

SceneEditor::eTriggerAction SceneEditor::get_trigger_action(const float delta_time)
{
    if (Input::is_trigger_pressed(HAND_RIGHT)) {
        time_pressed_storage += delta_time;
        if (time_pressed_storage > TIME_UNTIL_LONG_PRESS) {
            return TRIGGER_HOLDED;
        }
        return NO_TRIGGER_ACTION;
    }

    else if (Input::was_trigger_released(HAND_RIGHT) && time_pressed_storage < TIME_UNTIL_LONG_PRESS) {
        time_pressed_storage = 0.0f;
        return TRIGGER_TAPPED;
    }

    time_pressed_storage = 0.0f;

    return NO_TRIGGER_ACTION;
}
