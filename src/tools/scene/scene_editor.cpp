#include "scene_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/sculpt_instance.h"
#include "framework/nodes/spot_light_3d.h"
#include "framework/nodes/omni_light_3d.h"
#include "framework/nodes/directional_light_3d.h"
#include "framework/math/intersections.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_color.wgsl.gen.h"
#include "shaders/ui/ui_xr_panel.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

MeshInstance3D* intersection_mesh = nullptr;

uint32_t subdivisions = 16;

void SceneEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    main_scene = Engine::instance->get_main_scene();

    gizmo_3d.initialize(TRANSLATION_GIZMO, { 0.0f, 0.0f, 0.0f });

    init_ui();

    // debug

    SculptInstance* default_sculpt = new SculptInstance();
    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    rooms_renderer->get_raymarching_renderer()->set_current_sculpt(default_sculpt);

    /*intersection_mesh = new MeshInstance3D();
    intersection_mesh->add_surface(RendererStorage::get_surface("box"));
    intersection_mesh->scale(glm::vec3(0.01f));

    Material intersection_mesh_material;
    intersection_mesh_material.color = colors::CYAN;
    intersection_mesh_material.shader = RendererStorage::get_shader_from_source(shaders::mesh_color::source, shaders::mesh_color::path, intersection_mesh_material);
    intersection_mesh->set_surface_material_override(intersection_mesh->get_surface(0), intersection_mesh_material);

    main_scene->add_node(intersection_mesh);*/
}

void SceneEditor::clean()
{
    gizmo_3d.clean();

    BaseEditor::clean();
}

void SceneEditor::update(float delta_time)
{
    BaseEditor::update(delta_time);

    if (moving_node) {

        static_cast<Node3D*>(selected_node)->set_translation(Input::get_controller_position(HAND_RIGHT, POSE_AIM));

        if (Input::was_trigger_pressed(HAND_RIGHT)) {
            moving_node = false;
        }
    }

    update_gizmo(delta_time);

    if (xr_panel_3d) {

        // Update welcome screen following headset??

        glm::mat4x4 m(1.0f);
        glm::vec3 eye = renderer->get_camera_eye();
        glm::vec3 new_pos = eye + renderer->get_camera_front();

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
        // m = glm::translate(m, -glm::vec3(1.0f, 1.0f, 0.0f));

        xr_panel_3d->set_model(m);
        xr_panel_3d->update(delta_time);
    }
    else {
        xr_panel_2d->update(delta_time);
    }

    // debug

    //glm::vec3 ray_origin;
    //glm::vec3 ray_direction;

    //if (Renderer::instance->get_openxr_available())
    //{
    //    ray_origin = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    //    glm::mat4x4 select_hand_pose = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
    //    ray_direction = get_front(select_hand_pose);
    //}
    //else
    //{
    //    Camera* camera = Renderer::instance->get_camera();
    //    glm::vec3 ray_dir = camera->screen_to_ray(Input::get_mouse_position());

    //    ray_origin = camera->get_eye();
    //    ray_direction = glm::normalize(ray_dir);
    //}

    //// Quad
    //glm::mat4x4 model = curved_quad->get_model();

    //glm::vec3 quad_position = model[3];
    //glm::quat quad_rotation = glm::quat_cast(model);
    //glm::vec2 quad_size = { 2.0f, 1.0f };

    //float collision_dist;
    //glm::vec3 intersection_point;
    //glm::vec3 local_intersection_point;

    //if (intersection::ray_curved_quad(
    //    ray_origin,
    //    ray_direction,
    //    quad_position,
    //    quad_size * 0.5f,
    //    quad_rotation,
    //    subdivisions,
    //    0.25f,
    //    intersection_point,
    //    local_intersection_point,
    //    collision_dist,
    //    true
    //)) {
    //    intersection_mesh->set_translation(intersection_point);
    //    intersection_mesh->scale(glm::vec3(0.01f));
    //}
}

void SceneEditor::render()
{
    BaseEditor::render();

    RoomsEngine::render_controllers();

    render_gizmo();

    if (xr_panel_3d) {
        xr_panel_3d->render();
    }
    else {
        xr_panel_2d->render();
    }
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
            ui::ButtonSubmenu2D* lights_submenu = new ui::ButtonSubmenu2D("light", "data/textures/light.png");
            ui::ItemGroup2D* g_add_node = new ui::ItemGroup2D("g_light_types");
            g_add_node->add_child(new ui::TextureButton2D("omni", "data/textures/x.png"));
            g_add_node->add_child(new ui::TextureButton2D("spot", "data/textures/m.png"));
            g_add_node->add_child(new ui::TextureButton2D("directional", "data/textures/sun.png"));
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
        second_row->add_child(new ui::TextureButton2D("import", "data/textures/import.png"));
        second_row->add_child(new ui::TextureButton2D("export", "data/textures/export.png"));
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

    // Create tutorial/welcome panel
    {
        xr_panel_2d = new Node2D("tutorial_scene_root", { 0.0f, 0.0f }, { 1.0f, 1.0f });

        auto webgpu_context = Renderer::instance->get_webgpu_context();
        glm::vec2 size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height)) * 0.5f;
        glm::vec2 pos = size * 0.5f;

        if (renderer->get_openxr_available()) {
            size = glm::vec2(1920.f, 1080.0f);
            pos = -size * 0.5f;
        }

        ui::XRPanel* xr_panel = new ui::XRPanel("scene_editor_root", "data/images/welcome_screen.png", pos, size);
        xr_panel_2d->add_child(xr_panel);

        /*ui::HContainer2D* first_xr_panel_row = new ui::HContainer2D("first_xr_panel_row", { 0.0f, 0.0f });
        xr_panel->add_child(first_xr_panel_row);

        first_xr_panel_row->add_child(new ui::TextureButton2D("test_0", "data/textures/import.png"));
        first_xr_panel_row->add_child(new ui::TextureButton2D("test_1", "data/textures/export.png"));*/

        const glm::vec2& button_size = { size.x * 0.5f, size.y * 0.5 };

        //xr_panel->add_button({ button_size.x * 0.5f, size.y - button_size.y * 0.5f }, button_size);
        xr_panel->add_button({ size.x * 0.5f, size.y - button_size.y * 0.5f }, button_size);

        xr_panel->add_button({ button_size.x * 0.5f, button_size.y * 0.5f }, button_size);
        xr_panel->add_button({ size.x - button_size.x * 0.5f, button_size.y * 0.5f }, button_size);

        if (renderer->get_openxr_available()) {
            xr_panel_3d = new Viewport3D(xr_panel_2d);
            xr_panel_3d->set_active(true);
        }
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
        add_node(main_scene->get_nodes().back());
    });

    Node::bind("sculpt", [&](const std::string& signal, void* button) {
        SculptInstance* new_sculpt = new SculptInstance();
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        rooms_renderer->get_raymarching_renderer()->set_current_sculpt(new_sculpt);
        add_node(new_sculpt);
    });

    // Lights
    {
        Node::bind("omni", [&](const std::string& signal, void* button) { create_light_node(LIGHT_OMNI); });
        Node::bind("spot", [&](const std::string& signal, void* button) { create_light_node(LIGHT_SPOT); });
        Node::bind("directional", [&](const std::string& signal, void* button) { create_light_node(LIGHT_DIRECTIONAL); });
    }

    Node::bind("clone", [&](const std::string& signal, void* button) { clone_node(); });

    // Gizmo events

    Node::bind("move", [&](const std::string& signal, void* button) { set_gizmo_translation(); });
    Node::bind("rotate", [&](const std::string& signal, void* button) { set_gizmo_rotation(); });
    Node::bind("scale", [&](const std::string& signal, void* button) { set_gizmo_scale(); });
}

void SceneEditor::add_node(Node* node)
{
    selected_node = node;

    // To allow the user to move the node at the beginning
    moving_node = is_gizmo_usable() && renderer->get_openxr_available();
}

void SceneEditor::clone_node()
{
    if (!selected_node) {
        return;
    }

    // selected_node.clone() ?
}

void SceneEditor::create_light_node(uint8_t type)
{
    Light3D* new_light = nullptr;

    switch (type)
    {
    case LIGHT_OMNI:
        new_light = new OmniLight3D();
        new_light->set_name("omni_light");
        new_light->set_translation({ 1.0f, 1.f, 0.0f });
        new_light->set_range(5.0f);
        break;
    case LIGHT_SPOT:
        new_light = new SpotLight3D();
        new_light->set_name("spot_light");
        new_light->set_translation({ 0.0f, 1.f, 0.0f });
        new_light->rotate(glm::radians(-90.f), { 1.f, 0.0f, 0.f });
        new_light->set_range(5.0f);
        break;
        case LIGHT_DIRECTIONAL:
        new_light = new DirectionalLight3D();
        new_light->set_name("directional_light");
        new_light->rotate(glm::radians(-90.f), { 1.f, 0.0f, 0.f });
        break;
    default:
        assert(0 && "Unsppported light type!");
        break;
    }

    new_light->set_color({ 1.0f, 1.0f, 1.0f });
    new_light->set_intensity(1.0f);

    main_scene->add_node(new_light);
    add_node(new_light);
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