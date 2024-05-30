#include "scene_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/nodes/viewport_3d.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

Gizmo2D SceneEditor::gizmo_2d = {};
Gizmo3D SceneEditor::gizmo_3d = {};

void SceneEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    main_scene = Engine::instance->get_main_scene();

    gizmo_3d.initialize(TRANSLATION_GIZMO, { 0.0f, 0.0f, 0.0f });

    init_ui();
}

void SceneEditor::clean()
{
    gizmo_3d.clean();

    BaseEditor::clean();
}

void SceneEditor::update(float delta_time)
{
    BaseEditor::update(delta_time);

    update_gizmo(delta_time);
}

void SceneEditor::render()
{
    BaseEditor::render();

    RoomsEngine::render_controllers();

    render_gizmo(); 
}

void SceneEditor::render_gui()
{

}

void SceneEditor::init_ui()
{
    main_panel_2d = new ui::HContainer2D("scene_editor_root", { 64.0f, 64.f });

    // Color picker...

    {
        ui::ColorPicker2D* color_picker = new ui::ColorPicker2D("light_color_picker", colors::WHITE);
        color_picker->set_visibility(false);
        main_panel_2d->add_child(color_picker);
    }

    ui::VContainer2D* vertical_container = new ui::VContainer2D("scene_vertical_container", { 0.0f, 0.0f });
    main_panel_2d->add_child(vertical_container);

    // Add main rows
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    ui::HContainer2D* second_row = new ui::HContainer2D("row_1", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    // ** Clone node **
    // Note: This should be disabled if no node is selected
    first_row->add_child(new ui::TextureButton2D("clone", "data/textures/clone.png", 0/*ui::DISABLED*/));

    // ** Manipulate sculpt **
    {
        // Note: This should be disabled if no sculpt is selected
        first_row->add_child(new ui::TextureButton2D("edit_sculpt", "data/textures/cube_add.png", 0/*ui::DISABLED*/));
    }

    // ** Posible scene nodes **
    {
        ui::ButtonSubmenu2D* add_node_submenu = new ui::ButtonSubmenu2D("add_node", "data/textures/add.png");

        add_node_submenu->add_child(new ui::TextureButton2D("gltf", "data/textures/m.png"));
        add_node_submenu->add_child(new ui::TextureButton2D("sculpt", "data/textures/m.png"));

        // Lights
        {
            ui::ButtonSubmenu2D* lights_submenu = new ui::ButtonSubmenu2D("light", "data/textures/x.png");
            ui::ItemGroup2D* g_add_node = new ui::ItemGroup2D("g_light_types");
            g_add_node->add_child(new ui::TextureButton2D("omni", "data/textures/m.png"));
            g_add_node->add_child(new ui::TextureButton2D("directional", "data/textures/m.png"));
            g_add_node->add_child(new ui::TextureButton2D("spot", "data/textures/m.png"));
            lights_submenu->add_child(g_add_node);
            add_node_submenu->add_child(lights_submenu);
        }

        first_row->add_child(add_node_submenu);
    }

    // ** Gizmo modes **
    {
        ui::ComboButtons2D* combo_gizmo_modes = new ui::ComboButtons2D("combo_gizmo_modes");
        combo_gizmo_modes->add_child(new ui::TextureButton2D("move", "data/textures/translation_gizmo.png", ui::SELECTED));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("rotate", "data/textures/rotation_gizmo.png"));
        combo_gizmo_modes->add_child(new ui::TextureButton2D("scale", "data/textures/scale_gizmo.png"));
        second_row->add_child(combo_gizmo_modes);
    }

    // ** Import/Export scene **
    {
        second_row->add_child(new ui::TextureButton2D("import", "data/textures/x.png"));
        second_row->add_child(new ui::TextureButton2D("export", "data/textures/y.png"));
    }

    // ** Undo/Redo scene **
    {
        second_row->add_child(new ui::TextureButton2D("scene_undo", "data/textures/undo.png"));
        second_row->add_child(new ui::TextureButton2D("scene_redo", "data/textures/redo.png"));
    }

    if (renderer->get_openxr_available()) {
        main_panel_3d = new Viewport3D(main_panel_2d);
        main_panel_3d->set_active(true);
    }

    // Load controller UI labels
    //if (renderer->get_openxr_available())
    //{
    //    // Thumbsticks
    //    // Buttons
    //    // Triggers

    //    glm::vec2 double_size = { 2.0f, 1.0f };

    //    // Left hand
    //    {
    //        left_hand_container = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f });

    //        left_hand_container->add_child(new ui::ImageLabel2D("Round Shape", "data/textures/buttons/l_thumbstick.png", LAYOUT_ANY_NO_SHIFT_L));
    //        left_hand_container->add_child(new ui::ImageLabel2D("Smooth", "data/textures/buttons/l_grip_plus_l_thumbstick.png", LAYOUT_ANY_SHIFT_L, double_size));
    //        left_hand_container->add_child(new ui::ImageLabel2D("Redo", "data/textures/buttons/y.png", LAYOUT_ANY_NO_SHIFT_L));
    //        left_hand_container->add_child(new ui::ImageLabel2D("Guides", "data/textures/buttons/l_grip_plus_y.png", LAYOUT_ANY_SHIFT_L, double_size));
    //        left_hand_container->add_child(new ui::ImageLabel2D("Undo", "data/textures/buttons/x.png", LAYOUT_ANY_NO_SHIFT_L));
    //        left_hand_container->add_child(new ui::ImageLabel2D("PBR", "data/textures/buttons/l_grip_plus_x.png", LAYOUT_ANY_SHIFT_L, double_size));
    //        left_hand_container->add_child(new ui::ImageLabel2D("Manipulate Sculpt", "data/textures/buttons/l_trigger.png", LAYOUT_ALL));

    //        left_hand_ui_3D = new Viewport3D(left_hand_container);
    //        RoomsEngine::instance->get_main_scene()->add_node(left_hand_ui_3D);
    //    }

    //    // Right hand
    //    {
    //        right_hand_container = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f });

    //        right_hand_container->add_child(new ui::ImageLabel2D("Main size", "data/textures/buttons/r_thumbstick.png", LAYOUT_ANY_NO_SHIFT_R));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Sec size", "data/textures/buttons/r_grip_plus_r_thumbstick.png", LAYOUT_ANY_SHIFT_R, double_size));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Add/Substract", "data/textures/buttons/b.png", LAYOUT_SCULPT_NO_SHIFT_R));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Sculpt/Paint", "data/textures/buttons/r_grip_plus_b.png", LAYOUT_ANY_SHIFT_R, double_size));
    //        right_hand_container->add_child(new ui::ImageLabel2D("UI Select", "data/textures/buttons/a.png", LAYOUT_ALL));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Pick Material", "data/textures/buttons/r_grip_plus_a.png", LAYOUT_ANY_SHIFT_R, double_size));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Stamp", "data/textures/buttons/r_trigger.png", LAYOUT_ANY_NO_SHIFT_R));
    //        right_hand_container->add_child(new ui::ImageLabel2D("Smear", "data/textures/buttons/r_grip_plus_r_trigger.png", LAYOUT_ANY_SHIFT_R, double_size));

    //        right_hand_ui_3D = new Viewport3D(right_hand_container);
    //        RoomsEngine::instance->get_main_scene()->add_node(right_hand_ui_3D);
    //    }
    //}

    // Bind callbacks
    bind_events();
}

void SceneEditor::bind_events()
{
    Node::bind("edit_sculpt", [&](const std::string& signal, void* button) {
        RoomsEngine::switch_editor(SCULPT_EDITOR);
    });

    Node::bind("gltf", [&](const std::string& signal, void* button) {
        parse_scene("data/meshes/controllers/left_controller.glb", main_scene->get_nodes());
        selected_node = main_scene->get_nodes().back();
    });

    Node::bind("clone", [&](const std::string& signal, void* button) { clone_node(); });

    // Gizmo events

    Node::bind("move", [&](const std::string& signal, void* button) { set_gizmo_translation(); });
    Node::bind("rotate", [&](const std::string& signal, void* button) { set_gizmo_rotation(); });
    Node::bind("scale", [&](const std::string& signal, void* button) { set_gizmo_scale(); });
}

void SceneEditor::clone_node()
{
    if (!selected_node) {
        return;
    }

    // selected_node.clone() ?
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

    Node3D* node = static_cast<Node3D*>(selected_node);
    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    Transform t = mat4ToTransform(node->get_model());

    if (gizmo_3d.update(t, right_controller_pos, delta_time)) {
        node->set_transform(t);
    }
}

void SceneEditor::render_gizmo()
{
    if (!is_gizmo_usable()) {
        return;
    }

    if (renderer->get_openxr_available()) {
        gizmo_3d.render();
    }
    else {

        Camera* camera = renderer->get_camera();
        Node3D* node = static_cast<Node3D*>(selected_node);
        glm::mat4x4 m = node->get_model();

        if (gizmo_2d.render(camera->get_view(), camera->get_projection(), m)) {
            node->set_model(m);
        }
    }
}

void SceneEditor::set_gizmo_translation()
{
    if (renderer->get_openxr_available()) {
        gizmo_3d.set_operation(TRANSLATION_GIZMO);
    }
    else {
        gizmo_2d.set_operation(ImGuizmo::TRANSLATE);
    }
}

void SceneEditor::set_gizmo_rotation()
{
    if (renderer->get_openxr_available()) {
        gizmo_3d.set_operation(ROTATION_GIZMO);
    }
    else {
        gizmo_2d.set_operation(ImGuizmo::ROTATE);
    }
}

void SceneEditor::set_gizmo_scale()
{
    if (renderer->get_openxr_available()) {
        gizmo_3d.set_operation(SCALE_GIZMO);
    }
    else {
        gizmo_2d.set_operation(ImGuizmo::SCALE);
    }
}
