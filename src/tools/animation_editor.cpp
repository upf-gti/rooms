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

AnimationPlayer* player = nullptr;

uint64_t AnimationEditor::keyframe_signal_uid = 0;
uint64_t AnimationEditor::node_signal_uid = 0;

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

    player = new AnimationPlayer("Animation Player");

    init_ui();

    // Test the keyframe creation
    /*node->set_position({ 0.0f, 1.0f, 0.0 });
    process_keyframe();
    node->set_position({ 1.0f, 1.0f, 0.0 });
    process_keyframe();*/
}

void AnimationEditor::clean()
{
    
}

void AnimationEditor::on_enter(void* data)
{
    current_node = reinterpret_cast<Node3D*>(data);

    player->set_root_node(current_node);

    // Create animation for current node
    // TODO: Use the uuid for registering the animation
    std::string animation_name = current_node->get_name() + "@animation";
    current_animation = RendererStorage::get_animation(animation_name);

    if (!current_animation) {
        current_animation = new Animation();
        current_animation->set_name(animation_name);
        RendererStorage::register_animation(animation_name, current_animation);
    }
}

void AnimationEditor::update(float delta_time)
{
    if (adding_keyframe && Input::was_key_pressed(GLFW_KEY_ENTER)) {
        process_keyframe();
    }

    player->update(delta_time);

    if (current_node) {
        current_node->update(delta_time);
    }

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

    if (current_node) {
        current_node->render();
    }
}

void AnimationEditor::add_keyframe()
{
    // Get the current state of the animatable properties of the node
    store_animation_state(current_animation_properties);

    adding_keyframe = true;

    // Inspect useful data
    inspector->clear();

    inspect_keyframe_properties();
    inspect_node(current_node);

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector->set_visibility(true);
}

void AnimationEditor::process_keyframe()
{
    if (!current_node) {
        assert(0);
    }

    // Read the properties in order to see if there is any change
    sAnimationState new_anim_state;
    store_animation_state(new_anim_state);

    std::string* changed_properties = new std::string[new_anim_state.properties.size()];

    uint32_t changed_properties_count = get_changed_properties_from_states(current_animation_properties, new_anim_state, changed_properties);

    // Keyframe changes state
    if (changed_properties_count > 0u) {

        for (uint32_t i = 0u; i < changed_properties_count; i++) {

            std::string property_name = changed_properties[i];

            sAnimationState::sPropertyState& current_property_state = current_animation_properties.properties[property_name];

            if (current_property_state.track_id == -1) {
                // Add new track and set track id in struct
                std::cout << "Create new track on property " << property_name << std::endl;
                current_property_state.track_id = current_animation->get_track_count();

                current_track = current_animation->add_track(current_property_state.track_id);
                current_track->set_name(property_name);
                current_track->set_path(current_node->get_name() + "/" + property_name);
            }

            // Create and add keypoint to track
            std::cout << "Add keypoint to track " << property_name << std::endl;

            uint32_t num_keys = current_track->size();
            current_track->resize(num_keys + 1);

            Keyframe& frame = current_track->get_keyframe(num_keys);

            frame.time = current_time;
            frame.in = 0.0f;
            frame.value = current_property_state.value;
            frame.out = 0.0f;

            current_time += 0.5f;

            current_animation->recalculate_duration();

            // Update value
            current_property_state.value = new_anim_state.properties[property_name].value;
        }

    }

    adding_keyframe = false;

    delete[] changed_properties;
}

void AnimationEditor::store_animation_state(sAnimationState& state)
{
    if (!current_node) {
        assert(0);
    }

    const std::unordered_map<std::string, Node::AnimatableProperty>& properties = current_node->get_animatable_properties();

    for (auto prop_it : properties) {

        void* data = prop_it.second.property;

        switch (prop_it.second.property_type) {
        case Node::AnimatablePropertyType::INT8:
            state.properties[prop_it.first].value = *((int8_t*)data);
            break;
        case Node::AnimatablePropertyType::INT16:
            state.properties[prop_it.first].value = *((int16_t*)data);
            break;
        case Node::AnimatablePropertyType::INT32:
            state.properties[prop_it.first].value = *((int32_t*)data);
            break;
        case Node::AnimatablePropertyType::INT64:
            //state.properties[prop_it.first].value = *((int64_t*)data);
            break;
        case Node::AnimatablePropertyType::UINT8:
            state.properties[prop_it.first].value = *((uint8_t*)data);
            break;
        case Node::AnimatablePropertyType::UINT16:
            state.properties[prop_it.first].value = *((uint16_t*)data);
            break;
        case Node::AnimatablePropertyType::UINT32:
            state.properties[prop_it.first].value = *((uint32_t*)data);
            break;
        case Node::AnimatablePropertyType::UINT64:
            //state.properties[prop_it.first].value = *((uint64_t*)data);
            break;
        case Node::AnimatablePropertyType::FLOAT32:
            state.properties[prop_it.first].value = *((float*)data);
            break;
        case Node::AnimatablePropertyType::FLOAT64:
            //state.properties[prop_it.first].value = *((long float*)data);
            break;
        case Node::AnimatablePropertyType::IVEC2:
            state.properties[prop_it.first].value = *((glm::ivec2*)data);
            break;
        case Node::AnimatablePropertyType::UVEC2:
            state.properties[prop_it.first].value = *((glm::uvec2*)data);
            break;
        case Node::AnimatablePropertyType::FVEC2:
            state.properties[prop_it.first].value = *((glm::vec2*)data);
            break;
        case Node::AnimatablePropertyType::IVEC3:
            state.properties[prop_it.first].value = *((glm::ivec3*)data);
            break;
        case Node::AnimatablePropertyType::UVEC3:
            state.properties[prop_it.first].value = *((glm::uvec3*)data);
            break;
        case Node::AnimatablePropertyType::FVEC3:
            state.properties[prop_it.first].value = *((glm::vec3*)data);
            break;
        case Node::AnimatablePropertyType::IVEC4:
            state.properties[prop_it.first].value = *((glm::ivec4*)data);
            break;
        case Node::AnimatablePropertyType::UVEC4:
            state.properties[prop_it.first].value = *((glm::uvec4*)data);
            break;
        case Node::AnimatablePropertyType::FVEC4:
            state.properties[prop_it.first].value = *((glm::vec4*)data);
            break;
        case Node::AnimatablePropertyType::QUAT:
            state.properties[prop_it.first].value = *((glm::quat*)data);
            break;
        case Node::AnimatablePropertyType::UNDEFINED:
            break;
        }
    }
}

void AnimationEditor::render_gui()
{
    if (current_animation)
    {
        ImGui::Text("Animation %s", current_animation->get_name().c_str());
        ImGui::Text("Num Tracks %d", current_animation->get_track_count());

        if (current_track) {
            ImGui::Text("Num Keyframes %d", current_track->size());
        }

        if (current_animation && ImGui::Button("Play")) {
            player->play(current_animation);
        }
    }

    auto engine = dynamic_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->show_tree_recursive(player);
    engine->show_tree_recursive(current_node);
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
    inspector = new ui::Inspector({ .name = "inspector_root", .title = "Animation",.position = { 32.0f, 32.f } });

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
    Node::bind("add_keyframe", [&](const std::string& signal, void* button) { add_keyframe(); });
}

void AnimationEditor::inspect_keyframe()
{
    inspector->clear();

    std::string key_name = "Keyframe" + std::to_string(current_keyframe_idx);

    inspector->same_line();
    inspector->add_icon("data/textures/pattern.png");
    inspector->add_label("empty", key_name);
    inspector->end_line();

    inspect_keyframe_properties();

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    keyframe_dirty = false;
}

void AnimationEditor::inspect_keyframe_properties()
{
    inspector->add_label("empty", "Keyframe Properties");
    inspector->same_line();
    /*std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
    inspector->add_slider(signal, light->get_intensity(), 0.0f, 10.0f, 2);*/
    inspector->add_label("empty", "Interpolation", ui::SKIP_TEXT_RECT);
    inspector->end_line();
}

void AnimationEditor::inspect_node(Node* node)
{
    // Add identifier for signals
    std::string node_name = node->get_name();

    std::string signal = node_name + std::to_string(node_signal_uid++) + "_label";
    inspector->add_label(signal, node_name);

    // Inspect node properties

    const std::unordered_map<std::string, Node::AnimatableProperty>& properties = node->get_animatable_properties();

    void* data = nullptr;

    for (auto prop_it : properties) {

        signal = node_name + std::to_string(node_signal_uid++);
        data = prop_it.second.property;

        inspector->same_line();
        inspector->add_label("empty", prop_it.first, ui::SKIP_TEXT_RECT);

        switch (prop_it.second.property_type) {
        case Node::AnimatablePropertyType::INT8:
            break;
        case Node::AnimatablePropertyType::INT16:
            break;
        case Node::AnimatablePropertyType::INT32:
            break;
        case Node::AnimatablePropertyType::INT64:
            break;
        case Node::AnimatablePropertyType::UINT8:
            break;
        case Node::AnimatablePropertyType::UINT16:
            break;
        case Node::AnimatablePropertyType::UINT32:
            break;
        case Node::AnimatablePropertyType::UINT64:
            break;
        case Node::AnimatablePropertyType::FLOAT32:
            inspector->add_slider(signal, *((float*)data), (float*)data);
            break;
        case Node::AnimatablePropertyType::FLOAT64:
            break;
        case Node::AnimatablePropertyType::IVEC2:
            break;
        case Node::AnimatablePropertyType::UVEC2:
            break;
        case Node::AnimatablePropertyType::FVEC2:
            break;
        case Node::AnimatablePropertyType::IVEC3:
            break;
        case Node::AnimatablePropertyType::UVEC3:
            break;
        case Node::AnimatablePropertyType::FVEC3:
            break;
        case Node::AnimatablePropertyType::IVEC4:
            break;
        case Node::AnimatablePropertyType::UVEC4:
            break;
        case Node::AnimatablePropertyType::FVEC4:
            break;
        case Node::AnimatablePropertyType::QUAT:
            break;
        case Node::AnimatablePropertyType::UNDEFINED:
            break;
        }

        inspector->end_line();
    }

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

            Node::bind(signal, [&, n = current_node](const std::string& sg, void* data) {
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

    inspector_transform_dirty = !inspector->get_visibility();

    if (force) {
        inspector->set_visibility(true);
    }
}
