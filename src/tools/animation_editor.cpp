#include "animation_editor.h"

#include "framework/utils/utils.h"
#include "framework/input.h"
#include "framework/scene/parse_obj.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/nodes/animation_player.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/ui/inspector.h"
#include "framework/animation/track.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

Node3D* root = nullptr;
MeshInstance3D* node = nullptr;
AnimationPlayer* player = nullptr;

uint64_t AnimationEditor::keyframe_signal_uid = 0;

sAnimationState create_animation_state_from_node(const Node* scene_node)
{
    sAnimationState new_state;

    const std::unordered_map<std::string, Node::AnimatableProperty>& properties = scene_node->get_animatable_properties();

    for (auto prop_it : properties) {

        switch (prop_it.second.property_type) {
        case Node::AnimatablePropertyType::INT8:
            new_state.properties[prop_it.first] = { *((int8_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::INT16:
            new_state.properties[prop_it.first] = { *((int16_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::INT32:
            new_state.properties[prop_it.first] = { *((int32_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::INT64:
            //new_state.properties[prop_it.first] = { *((int64_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UINT8:
            new_state.properties[prop_it.first] = { *((uint8_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UINT16:
            new_state.properties[prop_it.first] = { *((uint16_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UINT32:
            new_state.properties[prop_it.first] = { *((uint32_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UINT64:
            //new_state.properties[prop_it.first] = { *((uint64_t*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::FLOAT32:
            new_state.properties[prop_it.first] = { *((float*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::FLOAT64:
            //new_state.properties[prop_it.first] = { *((long float*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::IVEC2:
            new_state.properties[prop_it.first] = { *((glm::ivec2*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UVEC2:
            new_state.properties[prop_it.first] = { *((glm::uvec2*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::FVEC2:
            new_state.properties[prop_it.first] = { *((glm::vec2*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::IVEC3:
            new_state.properties[prop_it.first] = { *((glm::ivec3*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UVEC3:
            new_state.properties[prop_it.first] = { *((glm::uvec3*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::FVEC3:
            new_state.properties[prop_it.first] = { *((glm::vec3*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::IVEC4:
            new_state.properties[prop_it.first] = { *((glm::ivec4*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UVEC4:
            new_state.properties[prop_it.first] = { *((glm::uvec4*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::FVEC4:
            new_state.properties[prop_it.first] = { *((glm::vec4*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::QUAT:
            new_state.properties[prop_it.first] = { *((glm::quat*)prop_it.second.property) };
            break;
        case Node::AnimatablePropertyType::UNDEFINED:
            break;
        }
    }

    return new_state;
}

uint32_t get_changed_properties_from_states(const sAnimationState& prev_state,
                                            const sAnimationState& current_state,
                                            std::string* changed_properties_list)
{
    uint32_t changed_properties_count = 0u;
    // Compare the properties of one state to another
    for (auto it_base : prev_state.properties) {
        if (it_base.second.value != current_state.properties.at(it_base.first).value) {
            changed_properties_list[changed_properties_count++] = it_base.first;
        }
    }

    return changed_properties_count;
}

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

    // Get the current state of the animateble properties of the node
    current_animation_properties = create_animation_state_from_node(node);

    // Test the keyframe creation
    /*node->set_position({ 0.0f, 1.0f, 0.0 });
    process_keyframe();
    node->set_position({ 1.0f, 1.0f, 0.0 });
    process_keyframe();*/
}

void AnimationEditor::clean()
{

}

void AnimationEditor::update(float delta_time)
{
    root->update(delta_time);

    BaseEditor::update(delta_time);

    // Update inspector for keyframes
    if(keyframe_dirty) {
        inspect_keyframe();
    }

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

void AnimationEditor::process_keyframe()
{
    // Read the properties in order to see if there is any change
    sAnimationState new_anim_state = create_animation_state_from_node(node);

    std::string* changed_properties = new std::string[new_anim_state.properties.size()];

    uint32_t changed_properties_count = get_changed_properties_from_states(current_animation_properties, new_anim_state, changed_properties);

    if (changed_properties_count > 0u) {
        // Keyframe changes state

        for (uint32_t i = 0u; i < changed_properties_count; i++) {
            if (current_animation_properties.properties[changed_properties[i]].track_id == -1) {
                // Add new track and set track id in struct
                std::cout << "Create new track on property " << changed_properties[i] << std::endl;
                current_animation_properties.properties[changed_properties[i]].track_id = 0u;
            }

            // Create, and add keypoint to track
            std::cout << "Add keypoint to track " << changed_properties[i] << std::endl;

            // Update value
            current_animation_properties.properties[changed_properties[i]].value = new_anim_state.properties[changed_properties[i]].value;
        }

    }

    delete[] changed_properties;
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
        inspect_keyframes_list(true);
    });

    Node::bind("play_animation", [&](const std::string& signal, void* button) {
        
    });

    // Keyframe actions events

    Node::bind("record_action", [&](const std::string& signal, void* button) { });
    Node::bind("add_keyframe", [&](const std::string& signal, void* button) { });
}

void AnimationEditor::inspect_keyframe()
{
    inspector->clear();

    std::string key_name = "Keyframe" + std::to_string(current_keyframe_idx);

    inspector->same_line();
    inspector->add_icon("data/textures/pattern.png");
    inspector->add_label("empty", key_name);
    inspector->end_line();

    // Keyframe properties
    {
        inspector->same_line();
        /*std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
        inspector->add_slider(signal, light->get_intensity(), 0.0f, 10.0f, 2);*/
        inspector->add_label("empty", "Interpolation");
        inspector->end_line();
        /*Node::bind(signal, [l = light](const std::string& sg, float value) {
            l->set_intensity(value);
        });*/
    }

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    keyframe_dirty = false;
}

void AnimationEditor::inspect_keyframes_list(bool force)
{
    inspector->clear();

    if (!current_track) {
        return;
    }

    for (uint32_t i = 0; i < current_track->size(); ++i) {
        Keyframe* key = &current_track->get_keyframe(i);

        inspector->same_line();

        // add unique identifier for signals
        std::string key_name = "Keyframe" + std::to_string(i);

        inspector->add_icon("data/textures/pattern.png");

        std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_label";
        inspector->add_label(signal, key_name);

        Node::bind(signal, [&, k = key, i](const std::string& sg, void* data) {
            current_keyframe_idx = i;
            current_keyframe = k;
            keyframe_dirty = true;
        });

        // Remove button
        {
            std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_remove";
            inspector->add_button(signal, "data/textures/delete.png");

            Node::bind(signal, [&, n = node](const std::string& sg, void* data) {
                // TODO
            });
        }

        inspector->end_line();
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
