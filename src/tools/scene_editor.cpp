#include "scene_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/parsers/parse_scene.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/spot_light_3d.h"
#include "framework/nodes/omni_light_3d.h"
#include "framework/nodes/directional_light_3d.h"
#include "framework/nodes/group_3d.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/math/intersections.h"
#include "framework/ui/inspector.h"
#include "framework/ui/keyboard.h"
#include "framework/camera/camera.h"
#include "framework/resources/room.h"
#include "framework/ui/io.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "graphics/managers/sculpt_manager.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <filesystem>

#define MAX_UNDO_STEPS 64

uint64_t SceneEditor::node_signal_uid = 0;

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
        inspector_dirty = true;
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

        SculptManager::sGPU_ReadResults* gpu_result = reinterpret_cast<SculptManager::sGPU_ReadResults*>(data);
        assert(gpu_result);
        const sGPU_SculptResults::sGPU_IntersectionData& intersection = gpu_result->loaded_results.ray_intersection;

        if (intersection.has_intersected == 0u) {
            return;
        }

        std::function<void(Node* node)> check_intersections = [&](Node* node) {
            SculptNode* sculpt_node = dynamic_cast<SculptNode*>(node);
            if (sculpt_node && sculpt_node->check_intersection(intersection.sculpt_id, intersection.instance_id)) {
                // hover the group
                if (!current_group && sculpt_node->get_parent()) {
                    hovered_node = sculpt_node->get_parent();
                    assert(dynamic_cast<Group3D*>(hovered_node));
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
        select_action_pressed = Input::was_trigger_pressed(HAND_RIGHT) || Input::was_mouse_released(GLFW_MOUSE_BUTTON_LEFT);

        if (Input::was_key_pressed(GLFW_KEY_ESCAPE)) {
            deselect();
        }
    }

    if (exports_dirty) {
        get_export_files();
    }

    update_hovered_node();

    shortcuts.clear();
    shortcuts[shortcuts::TOGGLE_SCENE_INSPECTOR] = true;

    if(!selected_node || (hovered_node != selected_node)) {
        shortcuts[shortcuts::SELECT_NODE] = true;
    }

    if (hovered_node) {
        process_node_hovered();
    }
    else if (moving_node) {

        assert(selected_node);

        shortcuts[shortcuts::PLACE_NODE] = true;

        static_cast<Node3D*>(selected_node)->set_position(Input::get_controller_position(HAND_RIGHT, POSE_AIM));

        if (Input::was_trigger_pressed(HAND_RIGHT)) {
            moving_node = false;
        }
    }

    if (Input::was_button_pressed(XR_BUTTON_Y) || Input::was_key_pressed(GLFW_KEY_I)) {
        inspector_dirty = true;
    }

    if (inspector_dirty) {

        if (selected_node && dynamic_cast<Light3D*>(selected_node)) {
            inspect_light();
        }
        else if (current_group) {
            inspect_group(true);
        }
        else {
            inspector_from_scene(true);
        }
    }

    update_gizmo(delta_time);

    update_node_transform();

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

void SceneEditor::on_enter(void* data)
{
    gizmo->set_operation(TRANSLATE);
    Node::emit_signal("combo_gizmo_modes@changed", (void*)"translate");
}

void SceneEditor::update_hovered_node()
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

void SceneEditor::process_node_hovered()
{
    const bool sculpt_hovered = !!dynamic_cast<SculptNode*>(hovered_node);
    const bool group_hovered = !sculpt_hovered && !!dynamic_cast<Group3D*>(hovered_node);
    const bool a_pressed = Input::was_button_pressed(XR_BUTTON_A);
    const bool b_pressed = Input::was_button_pressed(XR_BUTTON_B);

    if (current_group) {
        shortcuts[shortcuts::UNGROUP] = true;
        if (a_pressed) {
            ungroup_node(hovered_node, true, false);
        }
        else if (select_action_pressed) {
            select_node(hovered_node, false);
        }
    }
    else if (group_hovered) {
        if (grouping_node) {
            shortcuts[shortcuts::ADD_TO_GROUP] = true;
            if (a_pressed) {
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
    // In 2d, we have to select manually by click, so do not enter here!
    else if (grouping_node && renderer->get_openxr_available()) {
        shortcuts[shortcuts::CREATE_GROUP] = true;
        if (a_pressed) {
            process_group();
        }
    }
    else if (is_shift_right_pressed) {
        shortcuts[shortcuts::ANIMATE_NODE] = true;
        shortcuts[shortcuts::CLONE_NODE] = true;
        shortcuts[shortcuts::GROUP_NODE] = true;
        if (a_pressed) {
            clone_node(hovered_node, false);
        }
        else if (b_pressed) {
            selected_node = hovered_node;
            RoomsEngine::switch_editor(ANIMATION_EDITOR, hovered_node);
        }
        else if (select_action_pressed) {
            group_node(hovered_node);
        }
    }
    else {
        shortcuts[shortcuts::DUPLICATE_NODE] = true;
        shortcuts[shortcuts::EDIT_SCULPT_NODE] = sculpt_hovered;
        if (a_pressed) {
            clone_node(hovered_node, true);
        }
        else if (b_pressed && sculpt_hovered) {
            select_node(hovered_node, false);
            RoomsEngine::switch_editor(SCULPT_EDITOR, static_cast<SculptNode*>(hovered_node));
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

    main_panel = new ui::HContainer2D("scene_editor_root", { 48.0f, screen_size.y - 200.f }, ui::CREATE_3D);

    // Color picker...

    {
        ui::ColorPicker2D* color_picker = new ui::ColorPicker2D("light_color_picker", colors::WHITE);
        color_picker->set_visibility(false);
        main_panel->add_child(color_picker);
    }

    ui::VContainer2D* vertical_container = new ui::VContainer2D("scene_vertical_container", { 0.0f, 0.0f });
    main_panel->add_child(vertical_container);

    // Add main rows
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    ui::HContainer2D* second_row = new ui::HContainer2D("row_1", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    first_row->add_child(new ui::TextureButton2D("deselect", "data/textures/cross.png"));

    // ** Undo/Redo scene **
    {
        first_row->add_child(new ui::TextureButton2D("scene_undo", "data/textures/undo.png", ui::DISABLED));
        first_row->add_child(new ui::TextureButton2D("scene_redo", "data/textures/redo.png", ui::DISABLED));
    }

    // ** Node actions **
    {
        ui::ButtonSubmenu2D* node_actions_submenu = new ui::ButtonSubmenu2D("node_actions", "data/textures/cube.png", ui::DISABLED);
        node_actions_submenu->add_child(new ui::TextureButton2D("group", "data/textures/group.png"));
        node_actions_submenu->add_child(new ui::TextureButton2D("ungroup", "data/textures/ungroup.png"));
        node_actions_submenu->add_child(new ui::TextureButton2D("duplicate", "data/textures/clone.png"));
        node_actions_submenu->add_child(new ui::TextureButton2D("clone", "data/textures/clone_instance.png"));
        first_row->add_child(node_actions_submenu);
    }

    // ** Posible scene nodes **
    {
        ui::ButtonSubmenu2D* add_node_submenu = new ui::ButtonSubmenu2D("add_node", "data/textures/add.png");

        // add_node_submenu->add_child(new ui::TextureButton2D("gltf", "data/textures/monkey.png"));
        add_node_submenu->add_child(new ui::TextureButton2D("sculpt", "data/textures/sculpt.png"));

        // Lights
        {
            ui::ItemGroup2D* g_add_node = new ui::ItemGroup2D("g_light_types");
            g_add_node->set_visibility(false);
            g_add_node->add_child(new ui::TextureButton2D("omni", "data/textures/light.png"));
            g_add_node->add_child(new ui::TextureButton2D("spot", "data/textures/spot.png"));
            g_add_node->add_child(new ui::TextureButton2D("directional", "data/textures/sun.png"));
            add_node_submenu->add_child(g_add_node);
        }

        first_row->add_child(add_node_submenu);
    }

    // ** Display Settings **
    {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        ui::ButtonSubmenu2D* display_submenu = new ui::ButtonSubmenu2D("display", "data/textures/display_settings.png");
        display_submenu->set_visibility(false);
        ui::ItemGroup2D* g_display = new ui::ItemGroup2D("g_display");
        g_display->add_child(new ui::TextureButton2D("use_grid", "data/textures/grid.png", ui::ALLOW_TOGGLE | ui::SELECTED));
        g_display->add_child(new ui::TextureButton2D("use_environment", "data/textures/skybox.png", ui::ALLOW_TOGGLE | ui::SELECTED));
        g_display->add_child(new ui::FloatSlider2D("IBL_intensity", "data/textures/ibl_intensity.png", rooms_renderer->get_ibl_intensity(), ui::SliderMode::VERTICAL, ui::USER_RANGE/*ui::CURVE_INV_POW, 21.f, -6.0f*/, 0.0f, 4.0f, 2));
        display_submenu->add_child(g_display);
        display_submenu->add_child(new ui::FloatSlider2D("exposure", "data/textures/exposure.png", rooms_renderer->get_exposure(), ui::SliderMode::VERTICAL, ui::USER_RANGE/*ui::CURVE_INV_POW, 21.f, -6.0f*/, 0.0f, 4.0f, 2));
        first_row->add_child(display_submenu);
    }

    // ** Gizmo modes **
    {
        ui::ComboButtons2D* combo_gizmo_modes = new ui::ComboButtons2D("combo_gizmo_modes");
        combo_gizmo_modes->add_child(new ui::TextureButton2D("no_gizmo", "data/textures/no_gizmo.png"));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("translate", "data/textures/translation_gizmo.png", ui::SELECTED));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("rotate", "data/textures/rotation_gizmo.png"));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("scale", "data/textures/scale_gizmo.png"));
        second_row->add_child(combo_gizmo_modes);
    }

    // ** Import/Export scene **
    {
        second_row->add_child(new ui::TextureButton2D("load", "data/textures/load.png", ui::DISABLED));
        second_row->add_child(new ui::TextureButton2D("save", "data/textures/save.png", ui::DISABLED));
    }

    // ** Player stuff **
    {
        second_row->add_child(new ui::TextureButton2D("enter_room", "data/textures/play.png", ui::DISABLED));
    }

    // Create inspection panel (Nodes, properties, etc)
    {
        inspector = new ui::Inspector({ .name = "inspector_root", .title = "Scene Nodes",.position = {32.0f, 32.f}}, [&](ui::Inspector* scope) {
            return on_close_inspector();
        });
        inspector->set_visibility(false);
    }

    main_panel->set_visibility(false);

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
            left_hand_box->add_child(new ui::ImageLabel2D("Redo", shortcuts::L_GRIP_X_BUTTON_PATH, shortcuts::SCENE_REDO, double_size));
            left_hand_box->add_child(new ui::ImageLabel2D("Undo", shortcuts::X_BUTTON_PATH, shortcuts::SCENE_UNDO));
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            right_hand_box->add_child(new ui::ImageLabel2D("Edit Sculpt", shortcuts::B_BUTTON_PATH, shortcuts::EDIT_SCULPT_NODE));
            right_hand_box->add_child(new ui::ImageLabel2D("Edit Group", shortcuts::B_BUTTON_PATH, shortcuts::EDIT_GROUP));
            right_hand_box->add_child(new ui::ImageLabel2D("Animate", shortcuts::R_GRIP_B_BUTTON_PATH, shortcuts::ANIMATE_NODE, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Duplicate Node", shortcuts::A_BUTTON_PATH, shortcuts::DUPLICATE_NODE));
            right_hand_box->add_child(new ui::ImageLabel2D("Create Group", shortcuts::A_BUTTON_PATH, shortcuts::CREATE_GROUP));
            right_hand_box->add_child(new ui::ImageLabel2D("Add to Group", shortcuts::A_BUTTON_PATH, shortcuts::ADD_TO_GROUP));
            right_hand_box->add_child(new ui::ImageLabel2D("Ungroup", shortcuts::A_BUTTON_PATH, shortcuts::UNGROUP));
            right_hand_box->add_child(new ui::ImageLabel2D("Clone Node", shortcuts::R_GRIP_A_BUTTON_PATH, shortcuts::CLONE_NODE, double_size));
            right_hand_box->add_child(new ui::ImageLabel2D("Place Node", shortcuts::R_TRIGGER_PATH, shortcuts::PLACE_NODE));
            right_hand_box->add_child(new ui::ImageLabel2D("Select Node", shortcuts::R_TRIGGER_PATH, shortcuts::SELECT_NODE));
            right_hand_box->add_child(new ui::ImageLabel2D("Group Node", shortcuts::R_GRIP_R_TRIGGER_PATH, shortcuts::GROUP_NODE, double_size));
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
        main_scene->add_node(new_sculpt);
        select_node(new_sculpt);
        inspector_dirty = true;

        Node::emit_signal("@on_sculpt_added", (void*)nullptr);
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
    Node::bind("duplicate", [&](const std::string& signal, void* button) { clone_node(selected_node, true); });
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

bool SceneEditor::on_close_inspector()
{
    if (current_group) {
        current_group = nullptr;
        inspector_dirty = true;
        return false;
    }

    return true;
}

void SceneEditor::select_node(Node* node, bool place)
{
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
    if (!selected_node) {
        return nullptr;
    }

    RoomsEngine* engine = static_cast<RoomsEngine*>(RoomsEngine::instance);

    Node* new_node = NodeRegistry::get_instance()->create_node(node->get_node_type());
    node->clone(new_node, copy);

    SculptNode* sculpt_node = dynamic_cast<SculptNode*>(new_node);

    if (sculpt_node) {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        rooms_renderer->toogle_frame_debug();
        engine->set_current_sculpt(sculpt_node);
    }

    // Add to scene and select as current
    main_scene->add_node(new_node);
    select_node(new_node);

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_CLONE, node, new_node, copy });
    }

    inspector_dirty = true;

    return new_node;
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
    uintptr_t idx = reinterpret_cast<uintptr_t>(node);
    main_scene->remove_node(node);
    Node3D* node_3d = static_cast<Node3D*>(node);
    assert(node_3d);

    // TODO: Store correct index
    deleted_nodes[idx] = { -1, node_3d };

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_DELETE, node_3d }, false);
    }

    // Don't delete node now
    // delete node;

    inspector_dirty = true;
}

void SceneEditor::recover_node(Node* node, bool push_redo)
{
    uintptr_t idx = reinterpret_cast<uintptr_t>(node);

    if (!deleted_nodes.contains(idx)) {
        return;
    }

    const sDeletedNode& node_data = deleted_nodes[idx];
    Node3D* recovered_node = node_data.ref;
    main_scene->add_node(recovered_node, node_data.index);
    deleted_nodes.erase(idx);

    if (push_redo) {
        push_redo_action({ sActionData::ACTION_DELETE, recovered_node });
    }

    inspector_dirty = true;
}

void SceneEditor::ungroup_node(Node* node, bool push_undo, bool push_redo)
{
    if (dynamic_cast<Group3D*>(node)) {
        ungroup_all(node);
        return;
    }

    Node3D* g_node = static_cast<Node3D*>(node);
    Group3D* group = g_node->get_parent<Group3D*>();

    glm::mat4x4 world_space_model = g_node->get_global_model();
    group->remove_child(g_node);
    g_node->set_transform(Transform::mat4_to_transform(world_space_model));

    Scene* main_scene = Engine::instance->get_main_scene();
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

    inspector_dirty = true;
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

    // Check if current hover has group... (parent)
    Group3D* group = dynamic_cast<Group3D*>(hovered_3d);

    if (group) {
        // Add first node to the same group as the current hover
        main_scene->remove_node(to_group_3d);
        group->add_node(to_group_3d);
    }
    else {

        // Remove nodes from main scene
        main_scene->remove_node(to_group_3d);
        main_scene->remove_node(hovered_3d);

        // Create new group and add the nodes
        Group3D* new_group = new Group3D();
        new_group->add_nodes({ to_group_3d, hovered_3d });
        main_scene->add_node(new_group);
    }

    if (push_undo) {
        push_undo_action({ sActionData::ACTION_GROUP, to_group_3d });
    }

    grouping_node = false;
    inspector_dirty = true;
}

void SceneEditor::edit_group(Group3D* group)
{
    deselect();

    current_group = group;

    inspector_dirty = true;

    Node2D::get_widget_from_name<ui::TextureButton2D*>("group")->set_disabled(true);
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
        assert(0 && "Unsupported light type!");
        break;
    }

    new_light->set_color({ 1.0f, 1.0f, 1.0f });
    new_light->set_intensity(1.0f);
    new_light->set_range(5.0f);

    main_scene->add_node(new_light);
    select_node(new_light);
    inspector_dirty = true;
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

    if (current_group) {
        parent_transform = current_group->get_transform();
        t = Transform::combine(parent_transform, t);
    }

    if (gizmo->update(t, right_controller_pos, delta_time)) {

        if (!action_in_progress) {
            push_undo_action({ sActionData::ACTION_TRANSFORM, node, node->get_transform() });
            action_in_progress = true;
        }

        if (current_group) {
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

    if (current_group) {
        parent_transform = current_group->get_transform();
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

        if (current_group) {
            new_transform = Transform::combine(Transform::inverse(parent_transform), new_transform);
        }

        node->set_transform(new_transform);
    }
    else if (action_in_progress && !Input::is_mouse_pressed(GLFW_MOUSE_BUTTON_LEFT)) {
        action_in_progress = false;
    }
}

bool SceneEditor::is_rotation_being_used()
{
    return Input::get_trigger_value(HAND_LEFT) > 0.5;
}

void SceneEditor::update_node_transform()
{
    if (!selected_node) {
        return;
    }

    Node3D* node_3d = static_cast<Node3D*>(selected_node);

    // Do not rotate sculpt if shift -> we might be rotating the edit
    if (is_rotation_being_used() && !is_shift_left_pressed) {

        glm::quat left_hand_rotation = Input::get_controller_rotation(HAND_LEFT);
        glm::vec3 left_hand_translation = Input::get_controller_position(HAND_LEFT);
        glm::vec3 right_hand_translation = Input::get_controller_position(HAND_RIGHT);
        float hand_distance = glm::length2(left_hand_translation - right_hand_translation);

        if (!rotation_started) {
            last_left_hand_rotation = left_hand_rotation;
            last_left_hand_translation = left_hand_translation;
            push_undo_action({ sActionData::ACTION_TRANSFORM, node_3d, node_3d->get_transform() });
        }

        glm::quat rotation_diff = (left_hand_rotation * glm::inverse(last_left_hand_rotation));
        glm::vec3 translation_diff = left_hand_translation - last_left_hand_translation;

        node_3d->rotate_world(rotation_diff);
        node_3d->translate(translation_diff);

        if (Input::get_trigger_value(HAND_RIGHT) > 0.5) {

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

void SceneEditor::inspector_from_scene(bool force)
{
    inspector->clear(force, "Scene Nodes");

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
        inspector->button(signal, "data/textures/edit.png");

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
        });
    }

    if (flags & NODE_ANIMATE) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_animate";
        inspector->button(signal, "data/textures/animate.png");

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
        inspector->label(signal, node_name, flags);

        // Request keyboard and use the result to set the new node name. Not the nicest code, but anyway..
        {
            Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
                select_node(n, false);
            });

            auto callback = [&, n = node](const std::string& output) {
                n->set_name(output);
                inspector_dirty = true;
            };

            Node::bind(signal + "@long_click", [fn = callback, str = node_name](const std::string& sg, void* data) {
                ui::Keyboard::request(fn, str, 32u);
            });
        }
    }

    // Remove button
    if (flags & NODE_DELETE) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_remove";
        inspector->button(signal, "data/textures/delete.png", ui::CONFIRM_BUTTON);

        Node::bind(signal, [&, n = node](const std::string& sg, void* data) {

            if (selected_node == n) {
                deselect();
            }

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

    inspector->clear(!inspector->get_visibility(), "Group Nodes");

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

void SceneEditor::inspect_light()
{
    bool inspector_visible = inspector->get_visibility();
    inspector->clear(!inspector_visible);

    Light3D* light = static_cast<Light3D*>(selected_node);

    std::string node_name = selected_node->get_name();

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

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);
}

void SceneEditor::inspect_exports(bool force)
{
    bool inspector_visible = inspector->get_visibility();
    inspector->clear(!inspector_visible);

    for (const std::string& name : exported_scenes) {

        std::string full_name = "data/exports/" + name;

        inspector->same_line();

        std::string signal = name + std::to_string(node_signal_uid++) + "_load";
        inspector->button(signal, "data/textures/load.png");
        Node::bind(signal, [&, str = full_name](const std::string& sg, void* data) {
            deselect();
            static_cast<RoomsEngine*>(RoomsEngine::instance)->set_main_scene(str);
        });

        signal = name + std::to_string(node_signal_uid++) + "_add";
        inspector->button(signal, "data/textures/add.png");
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

    inspect_exports();

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
        recover_node(step.ref_node);
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
        recover_node(group, false);
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
        recover_node(group, false);
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
        recover_node(node_to_recover, false);
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
