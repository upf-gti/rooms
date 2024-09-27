#include "animation_editor.h"

#include "framework/utils/utils.h"
#include "framework/input.h"
#include "framework/scene/parse_obj.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/nodes/animation_player.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/ui/inspector.h"
#include "framework/animation/track.h"
#include "framework/math/math_utils.h"
#include "framework/camera/camera.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include "glm/gtx/quaternion.hpp"

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

    if (renderer->get_openxr_available()) {
        gizmo_3d.initialize(POSITION_SCALE_ROTATION_GIZMO);
    }
    else {
        gizmo_2d.set_operation(ImGuizmo::TRANSLATE | /*ImGuizmo::SCALE |*/ ImGuizmo::ROTATE);
    }

    init_ui();

    // Animation UI visualizations
    keyframe_markers_render_intstance = new MeshInstance3D();

    Material* joint_material = new Material();
    joint_material->set_depth_read(false);
    joint_material->set_priority(0);
    joint_material->set_transparency_type(ALPHA_BLEND);
    joint_material->set_color(glm::vec4(1.0f, 0.0f, 0.0f, 0.50f));
    joint_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path));

    keyframe_markers_render_intstance->set_frustum_culling_enabled(false);

    keyframe_markers_render_intstance->add_surface(RendererStorage::get_surface("sphere"));
    keyframe_markers_render_intstance->set_surface_material_override(keyframe_markers_render_intstance->get_surface(0), joint_material);

    // Trjacoetry line
    animation_trajectory_instance = new MeshInstance3D();
    animation_trajectory_mesh = new Surface();
    animation_trajectory_mesh->set_name("Animation trajecotry");
    animation_trajectory_instance->set_frustum_culling_enabled(false);

    InterleavedData empty;
    const std::vector<InterleavedData> empty_vertex = std::vector<InterleavedData>{ empty };
    animation_trajectory_mesh->update_vertex_buffer(empty_vertex);

    animation_trajectory_instance->add_surface(animation_trajectory_mesh);

    Material* skeleton_material = new Material();
    skeleton_material->set_color({ 1.0f, 0.0f, 0.0f, 1.0f });
    skeleton_material->set_depth_read(false);
    skeleton_material->set_priority(0);
    skeleton_material->set_topology_type(eTopologyType::TOPOLOGY_LINE_LIST);
    skeleton_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, skeleton_material));

    animation_trajectory_instance->set_surface_material_override(animation_trajectory_mesh, skeleton_material);
}

void AnimationEditor::clean()
{
    delete keyframe_markers_render_intstance;
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
        // Generate new animation
        current_animation = new Animation();
        current_animation->set_name(animation_name);
        RendererStorage::register_animation(animation_name, current_animation);

        animation_states.push_back({});

        sAnimationState& initial_state = animation_states.back();
        store_animation_state(initial_state);

        current_animation_state = &initial_state;

        // Generate a new track of every node property as a initial state

        for (auto& p : initial_state.properties) {

            sPropertyState& p_state = p.second;

            p_state.track_id = current_animation->get_track_count();

            current_track = current_animation->add_track(p_state.track_id);
            current_track->set_name(p.first);
            current_track->set_path(current_node->get_name() + "/" + p.first);
            current_track->set_type(current_track->get_type());

            // Store keyframe in property state
            p_state.keyframe = &current_track->add_keyframe({ .value = p_state.value, .in = 0.0f, .out = 0.0f, .time = 0.0f });
        }

        current_time += 0.5f;
        current_animation->recalculate_duration();
    }

    // Set inspector in front on VR mode
    if (renderer->get_openxr_available()) {

        glm::mat4x4 m(1.0f);
        glm::vec3 eye = dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_camera_eye();
        glm::vec3 new_pos = eye + dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_camera_front() * 0.6f;

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
        m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

        inspect_panel_3d->set_transform(Transform::mat4_to_transform(m));
        inspector_transform_dirty = false;
    }

    inspect_keyframes_list();

    update_animation_trajectory();
}

void AnimationEditor::update(float delta_time)
{
    player->update(delta_time);

    if (current_node) {
        current_node->update(delta_time);
    }

    BaseEditor::update(delta_time);

    update_gizmo(delta_time);

    // Update inspector for keyframes
    if(show_keyframe_dirty) {

        // Set current node in keyframe state
        bool is_looping = current_animation->get_looping();
        current_animation->set_looping(false);
        for (auto& p : current_animation_state->properties) {

            Node::AnimatableProperty node_property = current_node->get_animatable_property(p.first);
            void* data = node_property.property;

            current_animation->sample(current_animation_state->time, p.second.track_id, data);

            current_node->set_transform_dirty(true);

            // TODO: now the conversion void -> TYPE is done in the sample, but only supports 3 types
            // ...
        }

        current_animation->set_looping(is_looping);
        inspect_keyframe();
    }

    if (keyframe_dirty) {
        update_animation_trajectory();
    }

    if (renderer->get_openxr_available()) {

        if (!keyframe_dirty) {
            if (Input::was_button_pressed(XR_BUTTON_Y)) {
                inspect_keyframes_list(true);
            }
            else if (Input::was_button_pressed(XR_BUTTON_A)) {
                create_keyframe();
            }
        }
        // Creating keyframe
        else if (Input::was_button_pressed(XR_BUTTON_A)) {
            process_keyframe();
        }

        if (inspector_transform_dirty) {
            update_panel_transform();
        }

        inspect_panel_3d->update(delta_time);

        generate_shortcuts();
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

    render_gizmo();

    if (current_node) {
        current_node->render();
    }

    for (uint32_t i = 0u; i < animation_states.size(); i++) {
        const glm::vec3 position = std::get<glm::vec3>(animation_states[i].properties["translation"].value);

        const glm::mat4 anim_position_model = glm::scale(glm::translate(glm::mat4(1.0f), position), glm::vec3(0.006f));

        Renderer::instance->add_renderable(keyframe_markers_render_intstance, anim_position_model);
    }

    animation_trajectory_instance->render();
}

void AnimationEditor::render_gizmo()
{
    if (!keyframe_dirty || !current_node) {
        return;
    }

    if (renderer->get_openxr_available()) {
        gizmo_3d.render();
    }
    else {
        Camera* camera = renderer->get_camera();
        glm::mat4x4 m = current_node->get_model();

        if (gizmo_2d.render(camera->get_view(), camera->get_projection(), m)) {
            current_node->set_transform(Transform::mat4_to_transform(m));
        }
    }
}

void AnimationEditor::update_gizmo(float delta_time)
{
    if (!keyframe_dirty || !current_node) {
        return;
    }

    // Only 3D Gizmo for XR needs to update

    if (!renderer->get_openxr_available()) {
        return;
    }

    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    Transform t = current_node->get_transform();

    if (gizmo_3d.update(t, right_controller_pos, delta_time)) {
        current_node->set_transform(t);
    }
}

void AnimationEditor::update_animation_trajectory() {
    std::vector<InterleavedData> vertieces_to_upload;

    if (animation_states.size() <= 1u) {
        return;
    }

    for (uint32_t i = 0u; i < animation_states.size(); i++) {
        vertieces_to_upload.push_back({ std::get<glm::vec3>(animation_states[i].properties["translation"].value)});

        if (i + 1 >= animation_states.size()) {
            vertieces_to_upload.push_back({ std::get<glm::vec3>(animation_states[i].properties["translation"].value) });
        } else {
            vertieces_to_upload.push_back({ std::get<glm::vec3>(animation_states[i + 1].properties["translation"].value) });
        }
    }

    if (keyframe_dirty && current_node) {
        vertieces_to_upload.push_back(vertieces_to_upload.back());
        vertieces_to_upload.push_back({ current_node->get_translation()});
    }

    animation_trajectory_mesh->update_vertex_buffer(vertieces_to_upload);
}

/*
*   Opens the inspector to manipulate node and allow submitting/creating new keyframe
*/
void AnimationEditor::create_keyframe()
{
    // Get the last state to check changes later when adding new keyframes
    current_animation_state = &animation_states.back();

    keyframe_dirty = true;

    // Inspect useful data
    inspector->clear();

    inspect_keyframe_properties();

    inspect_node(current_node);

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector->set_visibility(true);

    // Manage other buttons
    {
        // Hide create_keyframe button
        auto w = Node2D::get_widget_from_name("create_keyframe");
        w->set_visibility(false);

        // Show submit_keyframe button
        w = Node2D::get_widget_from_name("submit_keyframe");
        w->set_visibility(true);

        w = Node2D::get_widget_from_name("open_list");
        static_cast<ui::Button2D*>(w)->set_disabled(true);
    }
}

/*
*   Adds a new keyframe or edits an existing one
*/

void AnimationEditor::process_keyframe()
{
    if (!current_node) {
        assert(0);
    }

    // Read the properties in order to see if there is any change
    
    sAnimationState new_anim_state;
    store_animation_state(new_anim_state);

    std::string* changed_properties = new std::string[new_anim_state.properties.size()];

    uint32_t changed_properties_count = get_changed_properties_from_states(*current_animation_state, new_anim_state, changed_properties);

    // Keyframe changes state
    if (changed_properties_count == 0u) {
        delete[] changed_properties;
        on_close();
        return;
    }

    for (uint32_t i = 0u; i < changed_properties_count; i++) {

        std::string property_name = changed_properties[i];

        sPropertyState& c_state = current_animation_state->properties[property_name];
        sPropertyState& n_state = new_anim_state.properties[property_name];
       
        current_track = current_animation->get_track_by_id(c_state.track_id);

        // Check if keyframe exists: modify value
        if (editing_keyframe && c_state.keyframe) {
            c_state.value = c_state.keyframe->value = n_state.value;
        }

        // No keyframe -> create one!
        else {
            // Create and add keypoint to track
            std::cout << "Add keypoint to track " << property_name << std::endl;

            // Create and update keyframe in the state
            if (editing_keyframe) {
                c_state.keyframe = &current_track->add_keyframe({ .value = n_state.value, .in = 0.0f, .out = 0.0f, .time = current_time });
            }
            else {
                n_state.keyframe = &current_track->add_keyframe({ .value = n_state.value, .in = 0.0f, .out = 0.0f, .time = current_time });
            }
        }
    }

    if (!editing_keyframe) {
        new_anim_state.time = current_time;
        current_time += 0.5f;
        current_animation->recalculate_duration();
        animation_states.push_back(new_anim_state);
    }

    delete[] changed_properties;
    on_close();
}

void AnimationEditor::store_animation_state(sAnimationState& state)
{
    if (!current_node) {
        assert(0);
    }

    const std::unordered_map<std::string, Node::AnimatableProperty>& properties = current_node->get_animatable_properties();

    for (auto prop_it : properties) {

        void* data = prop_it.second.property;

        // The tracks ids are always shared
        if (current_animation_state) {
            state.properties[prop_it.first].track_id = current_animation_state->properties[prop_it.first].track_id;
        }

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

void AnimationEditor::play_animation()
{
    player->play(current_animation);

    // Manage menu player buttons
    {
        // Hide play button
        auto w = Node2D::get_widget_from_name("play_animation");
        w->set_visibility(false);

        // Show stop button
        w = Node2D::get_widget_from_name("stop_animation");
        w->set_visibility(true);
    }
}

void AnimationEditor::stop_animation()
{
    player->stop(false);

    // Manage menu player buttons
    {
        // Show play button
        auto w = Node2D::get_widget_from_name("play_animation");
        w->set_visibility(true);

        // Hide stop button
        w = Node2D::get_widget_from_name("stop_animation");
        w->set_visibility(false);
    }
}

void AnimationEditor::render_gui()
{
    if (current_animation)
    {
        ImGui::Text("Animation %s", current_animation->get_name().c_str());
        ImGui::Text("Num Tracks %d", current_animation->get_track_count());
        ImGui::Text("Num States %zu", animation_states.size());

        if (current_animation && ImGui::Button("Play")) {
            player->play(current_animation);
        }
    }

    auto engine = dynamic_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->render_scene_tree_recursive(player);
    engine->render_scene_tree_recursive(current_node);
}

void AnimationEditor::init_ui()
{
    auto webgpu_context = Renderer::instance->get_webgpu_context();
    glm::vec2 screen_size = glm::vec2(static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height));

    main_panel_2d = new ui::HContainer2D("scene_editor_root", { 48.0f, screen_size.y - 136.f });

    // Add main row
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    main_panel_2d->add_child(first_row);

    // ** Go back to scene editor **
    first_row->add_child(new ui::TextureButton2D("go_back", "data/textures/back.png"));

    // ** Clone node **
    first_row->add_child(new ui::TextureButton2D("open_list", "data/textures/clone.png"));

    // ** Keyframe actions
    {
        ui::ItemGroup2D* g_keyframes = new ui::ItemGroup2D("g_keyframes");
        g_keyframes->add_child(new ui::TextureButton2D("record_action", "data/textures/record_action.png", ui::DISABLED));
        g_keyframes->add_child(new ui::TextureButton2D("create_keyframe", "data/textures/add_key.png"));
        g_keyframes->add_child(new ui::TextureButton2D("submit_keyframe", "data/textures/submit_key.png", ui::HIDDEN));
        first_row->add_child(g_keyframes);
    }

    // ** Play animation **
    first_row->add_child(new ui::TextureButton2D("play_animation", "data/textures/play.png"));
    first_row->add_child(new ui::TextureButton2D("stop_animation", "data/textures/stop.png", ui::HIDDEN));

    // Create inspection panel (Nodes, properties, etc)
    inspector = new ui::Inspector({ .name = "inspector_root", .title = "Animation",.position = { 32.0f, 32.f } }, [&](ui::Inspector* scope) {
        return on_close();
    });

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
            left_hand_box = new ui::VContainer2D("left_controller_root", { 0.0f, 0.0f });
            left_hand_box->add_child(new ui::ImageLabel2D("Keyframe List", shortcuts::Y_BUTTON_PATH, shortcuts::OPEN_KEYFRAME_LIST));
            left_hand_ui_3D = new Viewport3D(left_hand_box);
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f });
            right_hand_box->add_child(new ui::ImageLabel2D("Play Animation", shortcuts::B_BUTTON_PATH, shortcuts::PLAY_ANIMATION));
            right_hand_box->add_child(new ui::ImageLabel2D("Create keyframe", shortcuts::A_BUTTON_PATH, shortcuts::CREATE_KEYFRAME));
            right_hand_box->add_child(new ui::ImageLabel2D("Submit keyframe", shortcuts::A_BUTTON_PATH, shortcuts::SUBMIT_KEYFRAME));
            right_hand_ui_3D = new Viewport3D(right_hand_box);
        }
    }

    // Bind callbacks
    bind_events();
}

void AnimationEditor::bind_events()
{
    // Keyframe events
    {
        Node::bind("open_list", [&](const std::string& signal, void* button) { inspect_keyframes_list(true); });
        Node::bind("record_action", [&](const std::string& signal, void* button) { });
        Node::bind("create_keyframe", [&](const std::string& signal, void* button) { create_keyframe(); });
        Node::bind("submit_keyframe", [&](const std::string& signal, void* button) { if (keyframe_dirty) { process_keyframe(); } });
    }

    // Animation player events
    {
        Node::bind("play_animation", [&](const std::string& signal, void* button) { play_animation(); });
        Node::bind("stop_animation", [&](const std::string& signal, void* button) { stop_animation(); });
    }
}

void AnimationEditor::generate_shortcuts()
{
    std::unordered_map<uint8_t, bool> shortcuts;

    shortcuts[shortcuts::OPEN_KEYFRAME_LIST] = true;

    if (keyframe_dirty) {
        shortcuts[shortcuts::SUBMIT_KEYFRAME] = true;
    }
    else {
        shortcuts[shortcuts::CREATE_KEYFRAME] = true;
        shortcuts[shortcuts::PLAY_ANIMATION] = true;
    }

    BaseEditor::update_shortcuts(shortcuts);
}

void AnimationEditor::update_panel_transform()
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

void AnimationEditor::inspect_keyframe()
{
    inspector->clear();

    std::string key_name = "Keyframe";// +std::to_string(active_keyframe.idx);

    inspector->same_line();
    inspector->icon("data/textures/pattern.png");
    inspector->label("empty", key_name);
    inspector->end_line();

    inspect_keyframe_properties();
    inspect_node(current_node);

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    show_keyframe_dirty = false;

    auto w = Node2D::get_widget_from_name("open_list");
    static_cast<ui::Button2D*>(w)->set_disabled(true);
}

void AnimationEditor::inspect_keyframe_properties()
{
    inspector->label("empty", "Keyframe Properties");
    inspector->same_line();
    /*std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
    inspector->add_slider(signal, light->get_intensity(), 0.0f, 10.0f, 2);*/
    inspector->label("empty", "Interpolation", ui::SKIP_TEXT_RECT);
    inspector->end_line();
}

void AnimationEditor::inspect_node(Node* node)
{
    // Add identifier for signals
    std::string node_name = node->get_name();

    std::string signal = node_name + std::to_string(node_signal_uid++) + "_label";
    inspector->label(signal, node_name);

    // Inspect node properties

    const std::unordered_map<std::string, Node::AnimatableProperty>& properties = node->get_animatable_properties();

    void* data = nullptr;

    for (auto prop_it : properties) {

        signal = node_name + std::to_string(node_signal_uid++);
        data = prop_it.second.property;

        inspector->same_line();
        inspector->label("empty", prop_it.first, ui::SKIP_TEXT_RECT);

        switch (prop_it.second.property_type) {
        case Node::AnimatablePropertyType::INT8:
        case Node::AnimatablePropertyType::INT16:
        case Node::AnimatablePropertyType::INT32:
            inspector->islider(signal, *((int*)data), (int*)data);
            break;
        /*case Node::AnimatablePropertyType::INT64:
            break;*/
        /*case Node::AnimatablePropertyType::UINT8:
            break;
        case Node::AnimatablePropertyType::UINT16:
            break;
        case Node::AnimatablePropertyType::UINT32:
            break;*/
        /*case Node::AnimatablePropertyType::UINT64:
            break;*/
        case Node::AnimatablePropertyType::FLOAT32:
            inspector->fslider(signal, *((float*)data), (float*)data);
            break;
        /*case Node::AnimatablePropertyType::FLOAT64:
            break;*/
        case Node::AnimatablePropertyType::IVEC2:
            inspector->vector2<int>(signal, *((glm::ivec2*)data), 0, 64, (glm::ivec2*)data);
            break;
        /*case Node::AnimatablePropertyType::UVEC2:
            break;*/
        case Node::AnimatablePropertyType::FVEC2:
            inspector->vector2<float>(signal, *((glm::fvec2*)data), 0.0f, 1.0f, (glm::fvec2*)data);
            break;
        case Node::AnimatablePropertyType::IVEC3:
            inspector->vector3<int>(signal, *((glm::ivec3*)data), 0, 64, (glm::ivec3*)data);
            break;
        /*case Node::AnimatablePropertyType::UVEC3:
            break;*/
        case Node::AnimatablePropertyType::FVEC3:
            inspector->vector3<float>(signal, *((glm::fvec3*)data), 0.0f, 1.0f, (glm::fvec3*)data);
            // this is done everytime a vector3 is modified, move to another place!!
            Node::bind(signal + "@changed", [node = current_node](const std::string& signal, void* data) { node->set_transform_dirty(true); });
            break;
        case Node::AnimatablePropertyType::IVEC4:
            inspector->vector4<int>(signal, *((glm::ivec4*)data), 0, 64, (glm::ivec4*)data);
            break;
        /*case Node::AnimatablePropertyType::UVEC4:
            break;*/
        case Node::AnimatablePropertyType::FVEC4:
        case Node::AnimatablePropertyType::QUAT: // Using vector4 for showing quats
            if (prop_it.first.find("color") != std::string::npos) {
                inspector->color_picker(signal, *((Color*)data), (Color*)data);
            } else {
                inspector->vector4<float>(signal, *((glm::fvec4*)data), 0.0f, 1.0f, (glm::fvec4*)data);
            }
            break;
        case Node::AnimatablePropertyType::UNDEFINED:
            assert(0);
            break;
        }

        inspector->end_line();
    }
}

void AnimationEditor::inspect_keyframes_list(bool force)
{
    inspector->clear();

    for (uint32_t i = 0; i < animation_states.size(); ++i) {

        sAnimationState& s = animation_states[i];

        inspector->same_line();

        // add unique identifier for signals
        std::string key_name = "Keyframe" + std::to_string(i);

        inspector->icon("data/textures/pattern.png");

        std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_label";
        inspector->label(signal, key_name);

        Node::bind(signal, [&, it = i](const std::string& sg, void* data) {
            show_keyframe_dirty = true;
            keyframe_dirty = true;
            editing_keyframe = true;
            current_animation_state = &animation_states[it];
        });

        inspector->label("empty", std::to_string(s.time), ui::SKIP_TEXT_RECT);

        // Remove button
        {
            //std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_remove";
            //inspector->button(signal, "data/textures/delete.png");

            //Node::bind(signal, [&, n = current_node](const std::string& sg, void* data) {
            //    // TODO
            //});
        }

        inspector->end_line();
    }

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    // Enable xr for the buttons that need it..
    if (renderer->get_openxr_available()) {
        inspector->disable_2d();
    }

    inspector_transform_dirty = !inspector->get_visibility() || force;

    if (force) {
        inspector->set_visibility(true);
    }
}

bool AnimationEditor::on_close()
{
    bool should_close = !keyframe_dirty;

    if (!should_close) {
        inspect_keyframes_list();
    }

    // Hide submit_keyframe button
    auto w = Node2D::get_widget_from_name("submit_keyframe");
    w->set_visibility(false);

    // Show create_keyframe button
    w = Node2D::get_widget_from_name("create_keyframe");
    w->set_visibility(true);

    // Reactivate open list
    w = Node2D::get_widget_from_name("open_list");
    static_cast<ui::Button2D*>(w)->set_disabled(false);

    editing_keyframe = false;
    keyframe_dirty = false;
    current_animation_state = nullptr;

    update_animation_trajectory();

    return should_close;
}
