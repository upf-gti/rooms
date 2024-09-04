#include "animation_editor.h"

#include "framework/utils/utils.h"
#include "framework/input.h"
#include "framework/scene/parse_obj.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/nodes/animation_player.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/ui/inspector.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

Node3D* root = nullptr;
MeshInstance3D* node = nullptr;
AnimationPlayer* player = nullptr;

void AnimationEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    root = new Node3D();

    node = new MeshInstance3D();
    parse_obj("data/meshes/sphere.obj", node);
    node->set_name("node_0");

    Material* cube_material = new Material();
    cube_material->set_color(colors::PURPLE);
    cube_material->set_type(MATERIAL_PBR);
    cube_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path));

    node->set_surface_material_override(node->get_surface(0), cube_material);
    root->add_child(node);

    player = new AnimationPlayer("Animation Player");
    root->add_child(player);

    init_ui();
}

void AnimationEditor::clean()
{

}

void AnimationEditor::update(float delta_time)
{
    root->update(delta_time);

    BaseEditor::update(delta_time);

    if (renderer->get_openxr_available()) {

        if (inspector_transform_dirty) {
            // update_panel_transform();
        }

        inspect_panel_3d->update(delta_time);
    }
    else {
        inspector->update(delta_time);
    }
}

void AnimationEditor::render()
{
    RoomsEngine::render_controllers();

    BaseEditor::render();

    if (renderer->get_openxr_available()) {
        inspect_panel_3d->render();
    }
    else {
        inspector->render();
    }

    root->render();

    {
        static glm::mat4x4 test_model = node->get_model();
        Camera* camera = RoomsRenderer::instance->get_camera();
        bool changed = gizmo.render(camera->get_view(), camera->get_projection(), test_model);

        if (changed)
        {
            // node->set_model(test_model);
        }
    }
}

void AnimationEditor::render_gui()
{
    if (animation)
    {
        ImGui::Text("Animation %s", animation->get_name().c_str());
        ImGui::Text("Num Tracks %d", animation->get_track_count());

        if (ImGui::Button("Create keyframe")) {

            uint32_t num_keys = current_track->size();
            current_track->resize(num_keys + 1);

            Keyframe& frame = current_track->get_keyframe(num_keys);

            frame.time = current_time;

            frame.in = 0.0f;
            frame.value = node->get_translation();
            frame.out = 0.0f;

            current_time += 0.5f;

            animation->recalculate_duration();
        }

        if (current_track) {

            ImGui::Text("Num Keyframes %d", current_track->size());
            for (uint32_t i = 0; i < current_track->size(); ++i) {
                const Keyframe& key = current_track->get_keyframe(i);
                const glm::vec3& value = std::get<glm::vec3>(key.value);
                ImGui::Text("%d (%f) [%f %f %f]", i, key.time, value.x, value.y, value.z);
            }
        }
        else {
            if (ImGui::Button("Create position track")) {

                current_track = animation->add_track();
                current_track->set_name("translation");
                current_track->set_path(node->get_name() + "/" + "translation");
            }
        }

        if (ImGui::Button("Play")) {

            player->play("anim_test");
        }
    }
    else {
        if (ImGui::Button("Create animation")) {
            animation = new Animation();
            animation->set_name("anim_test");
            RendererStorage::register_animation(animation->get_name(), animation);
        }
    }

    auto engine = dynamic_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->show_tree_recursive(root);
}

void AnimationEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel_2d = new ui::HContainer2D("scene_editor_root", { 48.0f, screen_size.y - 136.f });

    // Add main row
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    main_panel_2d->add_child(first_row);

    // ** Clone node **
    first_row->add_child(new ui::TextureButton2D("toggle_list", "data/textures/clone.png"));

    // ** Keyframe actions
    {
        ui::ItemGroup2D* g_keyframes = new ui::ItemGroup2D("g_keyframes");
        g_keyframes->add_child(new ui::TextureButton2D("record_action", "data/textures/l.png"));
        g_keyframes->add_child(new ui::TextureButton2D("add_keyframe", "data/textures/add.png"));
        first_row->add_child(g_keyframes);
    }

    // ** Play animation **
    first_row->add_child(new ui::TextureButton2D("play_animation", "data/textures/a.png"));

    // Create inspection panel (Nodes, properties, etc)
    {
        inspector = new ui::Inspector({ .name = "inspector_root", .title = "Animation",.position = { 32.0f, 32.f } });
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
            // right_hand_container->add_child(new ui::ImageLabel2D("Scene Panel", "data/textures/buttons/b.png", LAYOUT_SCENE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Sculpt/Paint", "data/textures/buttons/r_grip_plus_b.png", LAYOUT_ANY_SHIFT_R, double_size));
            // right_hand_container->add_child(new ui::ImageLabel2D("Select Node", "data/textures/buttons/a.png", LAYOUT_SCENE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Pick Material", "data/textures/buttons/r_grip_plus_a.png", LAYOUT_ANY_SHIFT_R, double_size));
            // right_hand_container->add_child(new ui::ImageLabel2D("Place Node", "data/textures/buttons/r_trigger.png", LAYOUT_CLONE));
            // right_hand_container->add_child(new ui::ImageLabel2D("Make Instance", "data/textures/buttons/r_grip_plus_r_trigger.png", LAYOUT_CLONE_SHIFT, double_size));

            right_hand_ui_3D = new Viewport3D(right_hand_container);
        }
    }

    // Bind callbacks
    bind_events();
}

void AnimationEditor::bind_events()
{
    Node::bind("toggle_list", [&](const std::string& signal, void* button) {
        
    });

    Node::bind("play_animation", [&](const std::string& signal, void* button) {
        
    });

    // Keyframe actions events

    Node::bind("record_action", [&](const std::string& signal, void* button) { });
    Node::bind("add_keyframe", [&](const std::string& signal, void* button) { });
}
