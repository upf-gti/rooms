#include "scene_editor.h"

#include "includes.h"

#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/sculpt_instance.h"
#include "framework/nodes/spot_light_3d.h"
#include "framework/nodes/omni_light_3d.h"
#include "framework/nodes/directional_light_3d.h"
#include "framework/nodes/sculpt_instance.h"
#include "framework/math/intersections.h"
#include "framework/ui/inspector.h"
#include "framework/ui/keyboard.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/ui/ui_xr_panel.wgsl.gen.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <filesystem>

// MeshInstance3D* intersection_mesh = nullptr;
// uint32_t subdivisions = 16;

uint64_t SceneEditor::node_signal_uid = 0;

void SceneEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    main_scene = Engine::instance->get_main_scene();

    gizmo_3d.initialize(TRANSLATION_GIZMO, { 0.0f, 0.0f, 0.0f });

    init_ui();

#ifndef DISABLE_RAYMARCHER
    SculptInstance* default_sculpt = new SculptInstance();
    default_sculpt->set_name("default_sculpt");
    default_sculpt->initialize();
    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    rooms_renderer->get_raymarching_renderer()->set_current_sculpt(default_sculpt);
    main_scene->add_node(default_sculpt);
#endif

    Node::bind(main_scene->get_name() + "@nodes_added", [&](const std::string& sg, void* data) {
        set_inspector_dirty();
    });

    // debug
    /*intersection_mesh = new MeshInstance3D();
    intersection_mesh->add_surface(RendererStorage::get_surface("box"));
    intersection_mesh->scale(glm::vec3(0.01f));

    Material intersection_mesh_material;
    intersection_mesh_material.color = colors::CYAN;
    intersection_mesh_material.shader = RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, intersection_mesh_material);
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
    if (exports_dirty) {
        get_export_files();
    }

    if (inspector_dirty) {
        inspector_from_scene();
    }

    if (moving_node) {

        static_cast<Node3D*>(selected_node)->set_position(Input::get_controller_position(HAND_RIGHT, POSE_AIM));

        if (Input::was_trigger_pressed(HAND_RIGHT)) {
            moving_node = false;
        }
    }

    if (Input::was_button_pressed(XR_BUTTON_B)) {
        inspector_from_scene(true);
    }

    if (Input::was_key_pressed(GLFW_KEY_I)) {
        inspector_from_scene(true);
    }

    update_gizmo(delta_time);

    update_node_rotation();

    BaseEditor::update(delta_time);

    if (renderer->get_openxr_available()) {

        if (inspector_transform_dirty) {
            update_panel_transform();
        }

        inspect_panel_3d->update(delta_time);

        // Create current button layout based on state
        uint8_t current_layout = LAYOUT_SCENE;
        if (moving_node) current_layout = LAYOUT_CLONE;

        update_controller_flags(current_layout);
    }
    else {
        inspector->update(delta_time);
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
    //    intersection_mesh->set_position(intersection_point);
    //    intersection_mesh->scale(glm::vec3(0.01f));
    //}
}

void SceneEditor::render()
{
    RoomsEngine::render_controllers();

    render_gizmo();

    BaseEditor::render();

    if (renderer->get_openxr_available()) {
        inspect_panel_3d->render();
    } else {
        inspector->render();
    }
}

void SceneEditor::render_gui()
{

}

void SceneEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel_2d = new ui::HContainer2D("scene_editor_root", { 48.0f, screen_size.y - 200.f });

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

    // ** Undo/Redo scene **
    {
        // first_row->add_child(new ui::TextureButton2D("scene_undo", "data/textures/undo.png"));
        // first_row->add_child(new ui::TextureButton2D("scene_redo", "data/textures/redo.png"));
    }

    // ** Clone node **
    first_row->add_child(new ui::TextureButton2D("clone", "data/textures/clone.png"));

    // ** Posible scene nodes **
    {
        ui::ButtonSubmenu2D* add_node_submenu = new ui::ButtonSubmenu2D("add_node", "data/textures/add.png");

        // add_node_submenu->add_child(new ui::TextureButton2D("gltf", "data/textures/monkey.png"));
        add_node_submenu->add_child(new ui::TextureButton2D("sculpt", "data/textures/sculpt.png"));

        // Lights
        {
            ui::ItemGroup2D* g_add_node = new ui::ItemGroup2D("g_light_types");
            g_add_node->add_child(new ui::TextureButton2D("omni", "data/textures/light.png"));
            //g_add_node->add_child(new ui::TextureButton2D("spot", "data/textures/spot.png"));
            g_add_node->add_child(new ui::TextureButton2D("directional", "data/textures/sun.png"));
            add_node_submenu->add_child(g_add_node);
        }

        first_row->add_child(add_node_submenu);
    }

    // ** Display Settings **
    {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        ui::ItemGroup2D* g_display = new ui::ItemGroup2D("g_display");
        ui::ButtonSubmenu2D* display_submenu = new ui::ButtonSubmenu2D("display", "data/textures/display_settings.png");
        g_display->add_child(new ui::TextureButton2D("use_grid", "data/textures/grid.png", ui::ALLOW_TOGGLE | ui::SELECTED));
        g_display->add_child(new ui::TextureButton2D("use_environment", "data/textures/skybox.png", ui::ALLOW_TOGGLE | ui::SELECTED));
        g_display->add_child(new ui::Slider2D("IBL_intensity", "data/textures/ibl_intensity.png", rooms_renderer->get_ibl_intensity(), ui::SliderMode::VERTICAL, ui::USER_RANGE/*ui::CURVE_INV_POW, 21.f, -6.0f*/, 0.0f, 4.0f, 2));
        display_submenu->add_child(g_display);
        display_submenu->add_child(new ui::Slider2D("exposure", "data/textures/exposure.png", rooms_renderer->get_exposure(), ui::SliderMode::VERTICAL, ui::USER_RANGE/*ui::CURVE_INV_POW, 21.f, -6.0f*/, 0.0f, 4.0f, 2));
        first_row->add_child(display_submenu);
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
        second_row->add_child(new ui::TextureButton2D("load", "data/textures/load.png"));
        second_row->add_child(new ui::TextureButton2D("save", "data/textures/save.png"));
    }

    // Create inspection panel (Nodes, properties, etc)
    {
        inspector = new ui::Inspector({ .name = "inspector_root", .title = "Scene Nodes",.position = {32.0f, 32.f}});
        inspector->set_visibility(false);
    }

    if (renderer->get_openxr_available())
    {
        // create 3d viewports
        main_panel_3d = new Viewport3D(main_panel_2d);
        inspect_panel_3d = new Viewport3D(inspector);

        // Load controller UI labels

        // Thumbsticks
        // Buttons
        // Triggers

        glm::vec2 double_size = { 2.0f, 1.0f };

        // Left hand
        {
            left_hand_container = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f });

            // left_hand_container->add_child(new ui::ImageLabel2D("Round Shape", "data/textures/buttons/l_thumbstick.png", LAYOUT_ANY_NO_SHIFT_L));
            // left_hand_container->add_child(new ui::ImageLabel2D("Smooth", "data/textures/buttons/l_grip_plus_l_thumbstick.png", LAYOUT_ANY_SHIFT_L, double_size));
            // left_hand_container->add_child(new ui::ImageLabel2D("Redo", "data/textures/buttons/y.png", LAYOUT_ANY));
            // left_hand_container->add_child(new ui::ImageLabel2D("Guides", "data/textures/buttons/l_grip_plus_y.png", LAYOUT_ANY_SHIFT_L, double_size));
            // left_hand_container->add_child(new ui::ImageLabel2D("Undo", "data/textures/buttons/x.png", LAYOUT_ANY));
            // left_hand_container->add_child(new ui::ImageLabel2D("PBR", "data/textures/buttons/l_grip_plus_x.png", LAYOUT_ANY_SHIFT_L, double_size));
            // left_hand_container->add_child(new ui::ImageLabel2D("Manipulate Sculpt", "data/textures/buttons/l_trigger.png", LAYOUT_ALL));

            left_hand_ui_3D = new Viewport3D(left_hand_container);
        }

        // Right hand
        {
            right_hand_container = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f });

            // right_hand_container->add_child(new ui::ImageLabel2D("Main size", "data/textures/buttons/r_thumbstick.png", LAYOUT_ANY_NO_SHIFT_R));
            // right_hand_container->add_child(new ui::ImageLabel2D("Sec size", "data/textures/buttons/r_grip_plus_r_thumbstick.png", LAYOUT_ANY_SHIFT_R, double_size));
            right_hand_container->add_child(new ui::ImageLabel2D("Scene Panel", "data/textures/buttons/b.png", LAYOUT_SCENE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Sculpt/Paint", "data/textures/buttons/r_grip_plus_b.png", LAYOUT_ANY_SHIFT_R, double_size));
            right_hand_container->add_child(new ui::ImageLabel2D("Select Node", "data/textures/buttons/a.png", LAYOUT_SCENE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Pick Material", "data/textures/buttons/r_grip_plus_a.png", LAYOUT_ANY_SHIFT_R, double_size));
            right_hand_container->add_child(new ui::ImageLabel2D("Place Node", "data/textures/buttons/r_trigger.png", LAYOUT_CLONE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Make Instance", "data/textures/buttons/r_grip_plus_r_trigger.png", LAYOUT_CLONE_SHIFT, double_size));

            right_hand_ui_3D = new Viewport3D(right_hand_container);
        }
    }

    // Bind callbacks
    bind_events();
}

void SceneEditor::bind_events()
{
    Node::bind("gltf", [&](const std::string& signal, void* button) {
        parse_scene("data/meshes/controllers/left_controller.glb", main_scene->get_nodes());
        select_node(main_scene->get_nodes().back());
        inspector_dirty = true;
    });

    Node::bind("sculpt", [&](const std::string& signal, void* button) {
        SculptInstance* new_sculpt = new SculptInstance();
        new_sculpt->initialize();
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        rooms_renderer->toogle_frame_debug();
        rooms_renderer->get_raymarching_renderer()->set_current_sculpt(new_sculpt);
        main_scene->add_node(new_sculpt);
        select_node(new_sculpt);
        inspector_dirty = true;
    });

    // Environment / Scene Lights
    {
        Node::bind("omni", [&](const std::string& signal, void* button) { create_light_node(LIGHT_OMNI); });
        Node::bind("spot", [&](const std::string& signal, void* button) { create_light_node(LIGHT_SPOT); });
        Node::bind("directional", [&](const std::string& signal, void* button) { create_light_node(LIGHT_DIRECTIONAL); });

        Node::bind("exposure", [&](const std::string& signal, float value) {
            RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
            rooms_renderer->set_exposure(value);
        });

        Node::bind("IBL_intensity", [&](const std::string& signal, float value) {
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

    Node::bind("clone", [&](const std::string& signal, void* button) { clone_node(selected_node); });

    // Gizmo events

    Node::bind("move", [&](const std::string& signal, void* button) { set_gizmo_translation(); });
    Node::bind("rotate", [&](const std::string& signal, void* button) { set_gizmo_rotation(); });
    Node::bind("scale", [&](const std::string& signal, void* button) { set_gizmo_scale(); });

    // Export / Import (.room)
    {
        auto callback = [&](const std::string& output) {
            main_scene->serialize("data/exports/" + output + ".room");
            exports_dirty = true;
        };

        Node::bind("save", [fn = callback](const std::string& signal, void* button) { ui::Keyboard::request(fn, "unnamed"); });
        Node::bind("load", [&](const std::string& signal, void* button) { inspect_exports(true); });
    }
}

void SceneEditor::select_node(Node* node, bool place)
{
    selected_node = node;

    // To allow the user to move the node at the beginning
    moving_node = place && is_gizmo_usable() && renderer->get_openxr_available();

    if (!place && dynamic_cast<Light3D*>(node)) {
        inspect_light();
    }
}

void SceneEditor::clone_node(Node* node, bool copy)
{
    if (!selected_node) {
        return;
    }

    SculptInstance* current_sculpt = dynamic_cast<SculptInstance*>(node);

    // Only clone sculpt nodes by now
    if (current_sculpt == nullptr) {
        return;
    }

    SculptInstance* new_sculpt = nullptr;

    // raw copy, everything is recreated
    if (copy) {
        new_sculpt = new SculptInstance();
        new_sculpt->from_history(current_sculpt->get_stroke_history());
    }

    // instance copy, it should have different model, but uses same octree, etc.
    else {
        new_sculpt = new SculptInstance(current_sculpt);
    }

    new_sculpt->set_transform(current_sculpt->get_transform());
    new_sculpt->set_name(current_sculpt->get_name() + "_copy");

    // This should be used once we have the stuff to parent nodes
    //if (renderer->get_openxr_available()) {
    //    new_sculpt->set_position(Input::get_controller_position(HAND_RIGHT, POSE_AIM));
    //}

    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    rooms_renderer->toogle_frame_debug();
    rooms_renderer->get_raymarching_renderer()->set_current_sculpt(new_sculpt);

    // Add to scene and select as current
    main_scene->add_node(new_sculpt);
    select_node(new_sculpt);
    inspector_dirty = true;
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

    // Only 3D Gizmo for XR needs to update

    if (!renderer->get_openxr_available()) {
        return;
    }

    Node3D* node = static_cast<Node3D*>(selected_node);
    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    Transform t = node->get_transform();

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
            node->set_transform(Transform::mat4_to_transform(m));
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

void SceneEditor::update_panel_transform()
{
    glm::mat4x4 m(1.0f);
    glm::vec3 eye = renderer->get_camera_eye();
    glm::vec3 new_pos = eye + renderer->get_camera_front() * 0.6f;

    m = glm::translate(m, new_pos);
    m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
    m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

    inspect_panel_3d->set_transform(Transform::mat4_to_transform(m));

    inspector_transform_dirty = false;
}

bool SceneEditor::is_rotation_being_used()
{
    return Input::get_trigger_value(HAND_LEFT) > 0.5;
}

void SceneEditor::update_node_rotation()
{
    // Do not rotate sculpt if shift -> we might be rotating the edit
    if (selected_node && is_rotation_being_used() && !is_shift_left_pressed) {

        glm::quat current_hand_rotation = (Input::get_controller_rotation(HAND_LEFT));
        glm::vec3 current_hand_translation = Input::get_controller_position(HAND_LEFT);

        if (!rotation_started) {
            last_hand_rotation = current_hand_rotation;
            last_hand_translation = current_hand_translation;
        }

        glm::quat rotation_diff = (current_hand_rotation * glm::inverse(last_hand_rotation));
        glm::vec3 translation_diff = current_hand_translation - last_hand_translation;

        Node3D* node = static_cast<Node3D*>(selected_node);

        node->rotate_world(rotation_diff);
        node->translate(translation_diff);

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
}

void SceneEditor::inspector_from_scene(bool force)
{
    inspector->clear();

    auto& nodes = main_scene->get_nodes();

    for (auto node : nodes) {

        if (dynamic_cast<Light3D*>(node)) {
            inspect_node(node, NODE_LIGHT);
        }
        else if (dynamic_cast<SculptInstance*>(node)) {
            inspect_node(node, NODE_SCULPT);
        }
        else {
            inspect_node(node);
        }
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector_dirty = false;

    inspector_transform_dirty = !inspector->get_visibility();

    if (force) {
        inspector->set_visibility(true);
    }
}

void SceneEditor::inspect_node(Node* node, uint32_t flags, const std::string& texture_path)
{
    inspector->same_line();

    // add unique identifier for signals
    std::string node_name = node->get_name();

    if ((flags & NODE_ICON) && texture_path.size()) {
        inspector->add_icon(texture_path);
    }

    //if (flags & NODE_VISIBILITY) {
    //    std::string signal = node_name + std::to_string(node_signal_uid++) + "_visibility";
    //    inspector->add_button(signal, "data/textures/visibility.png", ui::ALLOW_TOGGLE);

    //    Node::bind(signal, [n = node](const std::string& sg, void* data) {
    //        // Implement visibility for Node3D
    //        // ...
    //        });
    //}

    if (flags & NODE_EDIT) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_edit";
        inspector->add_button(signal, "data/textures/edit.png");

        Node::bind(signal, [&, n = node, flags = flags](const std::string& sg, void* data) {

            select_node(n, false);

            // Set as current sculpt and go to sculpt editor
            if (dynamic_cast<SculptInstance*>(n)) {
                RoomsEngine::switch_editor(SCULPT_EDITOR);
                // TODO: do this in the on_enter of the sculpt editor passing the current node
                static_cast<RoomsEngine*>(RoomsEngine::instance)->set_current_sculpt(static_cast<SculptInstance*>(n));
            }
            else {
                // ...
            }
        });
    }

    if (flags & NODE_ANIMATE) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_animate";
        inspector->add_button(signal, "data/textures/animate.png");

        Node::bind(signal, [&, n = node, flags = flags](const std::string& sg, void* data) {

            selected_node = n;

            // TODO CHECK: This corrupts the pointer to the node "n"
            // select_node(n, false);

            RoomsEngine::switch_editor(ANIMATION_EDITOR, n);
        });
    }

    if (flags & NODE_NAME) {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_label";
        inspector->add_label(signal, node_name);

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
                ui::Keyboard::request(fn, str, 32);
            });
        }
    }

    // Remove button
    {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_remove";
        inspector->add_button(signal, "data/textures/delete.png");

        Node::bind(signal, [&, n = node](const std::string& sg, void* data) {

            if (selected_node == n) {
                selected_node = nullptr;
                moving_node = false;
            }

            main_scene->remove_node(n);

            delete n;

            inspector_dirty = true;
        });
    }

    inspector->end_line();
}

void SceneEditor::inspect_light()
{
    inspector->clear();

    Light3D* light = static_cast<Light3D*>(selected_node);

    std::string node_name = selected_node->get_name();

    inspector->same_line();
    inspector->add_icon("data/textures/light.png");
    inspector->add_label("empty", node_name);
    inspector->end_line();

    // Color
    {
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_picker";
        inspector->add_color_picker(signal, Color(light->get_color(), 1.0f));
        Node::bind(signal, [l = light](const std::string& sg, const Color& color) {
            l->set_color(color);
        });
    }

    // Intensity
    {
        inspector->same_line();
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
        inspector->add_slider(signal, light->get_intensity(), nullptr, 0.0f, 10.0f, 2);
        inspector->add_label("empty", "Intensity");
        inspector->end_line();
        Node::bind(signal, [l = light](const std::string& sg, float value) {
            l->set_intensity(value);
        });
    }

    // Range
    {
        inspector->same_line();
        std::string signal = node_name + std::to_string(node_signal_uid++) + "_range_slider";
        inspector->add_slider(signal, light->get_intensity(), nullptr, 0.0f, 5.0f, 2);
        inspector->add_label("empty", "Range");
        inspector->end_line();
        Node::bind(signal, [l = light](const std::string& sg, float value) {
            l->set_range(value);
        });
    }

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector->end_line();

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);
}

void SceneEditor::inspect_exports(bool force)
{
    inspector->clear();

    for (const std::string& name : exported_scenes) {

        std::string full_name = "data/exports/" + name;

        inspector->same_line();

        std::string signal = name + std::to_string(node_signal_uid++) + "_load";
        inspector->add_button(signal, "data/textures/load.png");
        Node::bind(signal, [&, str = full_name](const std::string& sg, void* data) {
            selected_node = nullptr;
            static_cast<RoomsEngine*>(RoomsEngine::instance)->set_main_scene(str);
        });

        signal = name + std::to_string(node_signal_uid++) + "_add";
        inspector->add_button(signal, "data/textures/add.png");
        Node::bind(signal, [&, str = full_name](const std::string& sg, void* data) {
            static_cast<RoomsEngine*>(RoomsEngine::instance)->add_to_main_scene(str);
        });

        inspector->add_label("empty", name);
        inspector->end_line();
    }

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector_transform_dirty = !inspector->get_visibility();

    if (force) {
        inspector->set_visibility(true);
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);
}

void SceneEditor::get_export_files()
{
    exported_scenes.clear();

    std::string path = "data/exports/";
    for (const auto& entry : std::filesystem::directory_iterator(path)) {

        std::string file_name = entry.path().string();
        std::string scene_name = file_name.substr(13);
        exported_scenes.push_back(scene_name);
    }

    inspect_exports();

    exports_dirty = false;
}
