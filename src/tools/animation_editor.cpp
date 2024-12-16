#include "animation_editor.h"

#include "framework/utils/utils.h"
#include "framework/input.h"
#include "framework/parsers/parse_obj.h"
#include "framework/parsers/parse_gltf.h"
#include "framework/nodes/skeleton_instance_3d.h"
#include "framework/nodes/joint_3d.h"
#include "framework/nodes/animation_player.h"
#include "framework/nodes/container_2d.h"
#include "framework/nodes/slider_2d.h"
#include "framework/nodes/button_2d.h"
#include "framework/nodes/group_3d.h"
#include "framework/nodes/character_3d.h"
#include "framework/ui/inspector.h"
#include "framework/animation/track.h"
#include "framework/animation/solvers/jacobian_solver.h"
#include "framework/math/math_utils.h"
#include "framework/camera/camera.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"

#include "engine/rooms_engine.h"

#include "tools/scene_editor.h"

#include "glm/gtx/quaternion.hpp"
#include "spdlog/spdlog.h"
#include "imgui.h"

AnimationPlayer* player = nullptr;

JacobianSolver ik_solver;
Transform ik_target;
Gizmo3D ik_gizmo;

uint64_t AnimationEditor::keyframe_signal_uid = 0;
uint64_t AnimationEditor::node_signal_uid = 0;

uint32_t get_changed_properties_from_states(const sAnimationState& prev_state,
                                            const sAnimationState& current_state,
                                            std::vector<std::string>& changed_properties_list)
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

    gizmo = static_cast<RoomsEngine*>(RoomsEngine::instance)->get_gizmo();

    init_ui();

    // Animation UI visualizations
    keyframe_markers_render_instance = new MeshInstance3D();

    Material* joint_material = new Material();
    joint_material->set_depth_read(false);
    joint_material->set_priority(0);
    joint_material->set_transparency_type(ALPHA_BLEND);
    joint_material->set_color(glm::vec4(1.0f, 0.0f, 0.0f, 0.50f));
    joint_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries));

    keyframe_markers_render_instance->set_frustum_culling_enabled(false);

    keyframe_markers_render_instance->add_surface(RendererStorage::get_surface("sphere"));
    keyframe_markers_render_instance->set_surface_material_override(keyframe_markers_render_instance->get_surface(0), joint_material);

    // Trajectory line
    animation_trajectory_instance = new MeshInstance3D();
    animation_trajectory_mesh = new Surface();
    animation_trajectory_mesh->set_name("Animation trajectory");
    animation_trajectory_instance->set_frustum_culling_enabled(false);

    const std::vector<glm::vec3> empty_vertex = { glm::vec3(0.0f)};
    animation_trajectory_mesh->update_vertex_buffer(empty_vertex);

    animation_trajectory_instance->add_surface(animation_trajectory_mesh);

    Material* skeleton_material = new Material();
    skeleton_material->set_color({ 1.0f, 0.0f, 0.0f, 1.0f });
    skeleton_material->set_depth_read(false);
    skeleton_material->set_priority(0);
    skeleton_material->set_topology_type(eTopologyType::TOPOLOGY_LINE_LIST);
    skeleton_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries, skeleton_material));

    animation_trajectory_instance->set_surface_material_override(animation_trajectory_mesh, skeleton_material);
}

void AnimationEditor::clean()
{
    BaseEditor::clean();

    delete keyframe_markers_render_instance;
    delete animation_trajectory_instance;
}

void AnimationEditor::on_enter(void* data)
{
    current_node = reinterpret_cast<Node3D*>(data);

    // Set the root always in the animation editor
    player->set_root_node(current_node);

    // Create animation for current node
    // TODO: Use the uuid for registering the animation
    std::string animation_name = current_node->get_name() + "@animation";
    current_animation = RendererStorage::get_animation(animation_name);

    // Search for any skeleton instance in the root
    auto skeleton_instance = find_skeleton(current_node);

    if (skeleton_instance) {
        current_node = skeleton_instance;
        // initialize_ik();
    }

    if (!current_animation) {

        current_time = 0.0f;

        // Generate new animation
        current_animation = new Animation();
        current_animation->set_name(animation_name);
        RendererStorage::register_animation(animation_name, current_animation);

        sAnimationState initial_state;
        store_animation_state(initial_state);

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

        sAnimationData new_anim_data;
        new_anim_data.animation = current_animation;
        new_anim_data.states.push_back(initial_state);
        animations_data[get_animation_idx()] = new_anim_data;

        current_time += 0.5f;
        current_animation->recalculate_duration();
    }
    else {
        // Start with last keyframe added time
        auto& data = animations_data[get_animation_idx()];
        current_time = data.current_time;
    }

    // Set inspector in front on VR mode
    if (renderer->get_openxr_available()) {

        glm::mat4x4 m(1.0f);
        glm::vec3 eye = static_cast<RoomsRenderer*>(Renderer::instance)->get_camera_eye();
        glm::vec3 new_pos = eye + static_cast<RoomsRenderer*>(Renderer::instance)->get_camera_front() * 0.6f;

        m = glm::translate(m, new_pos);
        m = m * glm::toMat4(get_rotation_to_face(new_pos, eye, { 0.0f, 1.0f, 0.0f }));
        m = glm::rotate(m, glm::radians(180.f), { 1.0f, 0.0f, 0.0f });

        inspector->set_xr_transform(Transform::mat4_to_transform(m));
        inspector_transform_dirty = false;
    }

    inspect_keyframes_list();

    update_animation_trajectory();

    gizmo->set_operation(TRANSLATE | ROTATE);
}

void AnimationEditor::on_exit()
{
    current_node = nullptr;
    current_joint = nullptr;

    stop_animation();

    // Store current time..
    auto& data = animations_data[get_animation_idx()];
    data.current_time = current_time;
}

void AnimationEditor::update(float delta_time)
{
    player->update(delta_time);

    BaseEditor::update(delta_time);

    // Update inspector for keyframes
    if(show_keyframe_dirty) {
        update_node_from_state(*current_animation_state);
        inspect_keyframe();
    }

    update_gizmo(delta_time);

    update_node_transform();

    // update_ik();

    if (keyframe_dirty) {
        update_animation_trajectory();
    }

    if (keyframe_list_dirty) {
        inspect_keyframes_list(true);
        keyframe_list_dirty = false;
    }

    // Get joints from skeleton
    bool select_pressed = Input::was_trigger_pressed(HAND_RIGHT) || Input::was_mouse_pressed(GLFW_MOUSE_BUTTON_LEFT);
    if(select_pressed &&
        (dynamic_cast<SkeletonInstance3D*>(current_node) || dynamic_cast<Character3D*>(current_node))) {
        glm::vec3 ray_origin;
        glm::vec3 ray_direction;
        float distance = 1e9f;

        Engine::instance->get_scene_ray(ray_origin, ray_direction);

        // This sets current_joint to a valid joint in the skeleton instance
        if (current_node->test_ray_collision(ray_origin, ray_direction, distance, reinterpret_cast<Node3D**>(&current_joint))) {
            inspector->clear();
            inspect_node(current_joint);
        }
    }

    if (renderer->get_openxr_available()) {

        if (!keyframe_dirty) {
            if (Input::was_button_pressed(XR_BUTTON_X)) {
                play_animation();
            }
            else if (Input::was_button_pressed(XR_BUTTON_Y)) {
                inspect_keyframes_list(true);
            }
            else if (Input::was_button_pressed(XR_BUTTON_A)) {
                create_keyframe();
            }
            else if (Input::was_button_pressed(XR_BUTTON_B)) {
                RoomsEngine::switch_editor(SCENE_EDITOR);
            }
        }
        // Creating keyframe
        else if (Input::was_button_pressed(XR_BUTTON_A)) {
            process_keyframe();
        }

        if (inspector_transform_dirty) {
            update_panel_transform();
        }

        generate_shortcuts();
    }

    inspector->update(delta_time);
}

void AnimationEditor::render()
{
    RoomsEngine::render_controllers();

    BaseEditor::render();

    inspector->render();

    render_gizmo();

    // render_ik();

    if (current_node) {
        current_node->render();
    }

    auto& states = animations_data[get_animation_idx()].states;

    for (uint32_t i = 0u; i < states.size(); i++) {
        const glm::vec3& position = std::get<glm::vec3>(states[i].properties["translation"].value);
        const glm::mat4& anim_position_model = glm::scale(glm::translate(glm::mat4(1.0f), position), glm::vec3(0.006f));
        Renderer::instance->add_renderable(keyframe_markers_render_instance, anim_position_model);
    }

    animation_trajectory_instance->render();
}

void AnimationEditor::update_gizmo(float delta_time)
{
    Node3D* node = get_current_node();

    // Gizmo only needs to update for XR
    if (!keyframe_dirty || !current_node || !renderer->get_openxr_available()) {
        return;
    }

    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    Transform t = node->get_global_transform();

    if (gizmo->update(t, right_controller_pos, delta_time)) {

        node->set_global_transform(t);

        if (dynamic_cast<Joint3D*>(node)) {
            current_node->set_transform_dirty(true);
        }
    }
}

void AnimationEditor::render_gizmo()
{
    Node3D* node = get_current_node();

    if (!keyframe_dirty || !node) {
        return;
    }

    gizmo->set_transform(node->get_global_transform());

    // This is only for 2D since Gizmo.render will only return true if
    // Gizmo2D is used!

    bool transform_dirty = gizmo->render();

    if (!transform_dirty || renderer->get_openxr_available()) {
        return;
    }

    node->set_global_transform(gizmo->get_transform());

    if (dynamic_cast<Joint3D*>(node)) {
        // Current node is skeletoninstance
        current_node->set_transform_dirty(true);
    }
}

Node3D* AnimationEditor::get_current_node()
{
    if (current_joint) {
        return current_joint;
    }
    return current_node;
}

SkeletonInstance3D* AnimationEditor::find_skeleton(Node* node)
{
    if (dynamic_cast<SkeletonInstance3D*>(node)) {
        return static_cast<SkeletonInstance3D*>(node);
    }
    for (auto child : node->get_children()) {
        auto instance = find_skeleton(child);
        if (dynamic_cast<SkeletonInstance3D*>(instance)) {
            return instance;
        }
    }
    return nullptr;
}

void AnimationEditor::initialize_ik()
{
    // Do left arm chain
    std::vector<Transform> chain;

    auto instance = static_cast<SkeletonInstance3D*>(current_node);

    // Origin joint in global space
    Pose& pose = instance->get_skeleton()->get_current_pose();
    chain.push_back(pose.get_global_transform(6u));
    // Rest of the joints in local space
    chain.push_back(pose.get_local_transform(8u));
    chain.push_back(pose.get_local_transform(10u));

    ik_solver.resize(3u);
    ik_solver.set_local_transform(0u, chain[0], 6u); // left arm
    ik_solver.set_local_transform(1u, chain[1], 8u); // "" forearm
    ik_solver.set_local_transform(2u, chain[2], 10u); // "" hand
    ik_solver.set_rotation_axis();

    // Set the target to the global position of the last joint of the chain
    ik_target.set_position(pose.get_global_transform(10u).get_position());
}

void AnimationEditor::update_ik()
{
    auto instance = dynamic_cast<SkeletonInstance3D*>(current_node);
    if (!instance || !keyframe_dirty) {
        return;
    }

    // Solve IK for character chain depending of the selected solver type
    ik_solver.solve(ik_target);

    // Update skeleton and current pose with the IK chain transforms
    // 1. Get origin joint in local space: Combine the inverse global transformation of its parent with its computed IK transformation
    const std::vector<uint32_t>& joint_idx = ik_solver.get_joint_indices();

    Pose& pose = instance->get_skeleton()->get_current_pose();
    Transform world_parent = pose.get_global_transform(pose.get_parent(joint_idx[0]));
    Transform world_child = ik_solver.get_local_transform(0);
    Transform local_child = Transform::combine(Transform::inverse(world_parent), world_child);
    // 2. Set the local transformation of the origin joint to the current pose
    pose.set_local_transform(joint_idx[0], local_child);
    // 3. For the rest of the chain, set the local transformation of each joint into the corresponding current pose joint
    for (uint32_t i = 1; i < joint_idx.size(); i++) {
        pose.set_local_transform(joint_idx[i], ik_solver.get_local_transform(i));
    }

    instance->update_joints_from_pose();
}

void AnimationEditor::render_ik()
{
    auto instance = dynamic_cast<SkeletonInstance3D*>(current_node);
    if (!instance || !keyframe_dirty) {
        return;
    }

    // Render gizmo for updating the target
    {
        Transform global = instance->get_global_transform();
        ik_gizmo.set_transform(Transform::combine(global, ik_target));

        if (ik_gizmo.render()) {
            ik_target = Transform::combine(Transform::inverse(global), ik_gizmo.get_transform());
        }
    }
}

uint32_t AnimationEditor::get_animation_idx()
{
    return reinterpret_cast<uintptr_t>(current_animation);
}

void AnimationEditor::update_node_from_state(const sAnimationState& state)
{
    // Set current node in keyframe state

    for (auto& p : state.properties) {

        Node::AnimatableProperty node_property = current_node->get_animatable_property(p.first);
        void* data = node_property.property;

        current_animation->sample(state.time, p.second.track_id, ANIMATION_LOOP_NONE, data, eInterpolationType::STEP);

        // TODO: now the conversion void -> TYPE is done in the sample, but only supports 3 types
        // ...
    }

    auto skeleton_instance = find_skeleton(current_node);
    if (skeleton_instance) {
        skeleton_instance->update_pose_from_joints();
    }
    else {
        current_node->set_transform_dirty(true);
    }
}

void AnimationEditor::update_node_transform()
{
    Node3D* node = get_current_node();

    if (!node) {
        return;
    }

    // Do not rotate sculpt if shift -> we might be rotating the edit
    if ((Input::get_trigger_value(HAND_LEFT) > 0.5f)) {

        Transform global_transform = node->get_global_transform();

        glm::quat left_hand_rotation = Input::get_controller_rotation(HAND_LEFT);
        glm::vec3 left_hand_translation = Input::get_controller_position(HAND_LEFT);
        glm::vec3 right_hand_translation = Input::get_controller_position(HAND_RIGHT);
        float hand_distance = glm::length2(left_hand_translation - right_hand_translation);

        if (!rotation_started) {
            last_left_hand_rotation = left_hand_rotation;
            last_left_hand_translation = left_hand_translation;
        }

        glm::quat rotation_diff = (left_hand_rotation * glm::inverse(last_left_hand_rotation));
        global_transform.rotate_world(rotation_diff);

        if (is_shift_left_pressed) {
            glm::vec3 translation_diff = left_hand_translation - last_left_hand_translation;
            global_transform.translate(translation_diff);
        }

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

        node->set_global_transform(global_transform);

        if (dynamic_cast<Joint3D*>(node)) {
            current_node->set_transform_dirty(true);
        }

        rotation_started = true;

        last_left_hand_rotation = left_hand_rotation;
        last_left_hand_translation = left_hand_translation;
    }

    // If rotation has stopped
    else if (rotation_started) {
        rotation_started = false;
    }
}

void AnimationEditor::update_animation_trajectory()
{
    std::vector<glm::vec3> vertices_to_upload;

    auto& states = animations_data[get_animation_idx()].states;

    if (states.size() <= 1u) {
        return;
    }

    for (uint32_t i = 0u; i < states.size(); i++) {
        vertices_to_upload.push_back(std::get<glm::vec3>(states[i].properties["translation"].value));

        if (i + 1 >= states.size()) {
            vertices_to_upload.push_back(std::get<glm::vec3>(states[i].properties["translation"].value));
        }
        else {
            vertices_to_upload.push_back(std::get<glm::vec3>(states[i + 1].properties["translation"].value));
        }
    }

    if (keyframe_dirty && current_node) {
        vertices_to_upload.push_back(vertices_to_upload.back());
        vertices_to_upload.push_back({ current_node->get_translation()});
    }

    animation_trajectory_mesh->update_vertex_buffer(vertices_to_upload);
}

/*
*   Opens the inspector to manipulate node and allow submitting/creating new keyframe
*/
void AnimationEditor::create_keyframe()
{
    auto& states = animations_data[get_animation_idx()].states;

    // Get the last state to check changes later when adding new keyframes
    current_animation_state = &states.back();
    update_node_from_state(*current_animation_state);

    keyframe_dirty = true;

    // Inspect useful data
    inspector->clear();
    inspector->set_visibility(true);

    inspect_keyframe_properties();
    inspect_node(get_current_node());

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

    std::vector<std::string> changed_properties;
    changed_properties.resize(new_anim_state.properties.size());

    uint32_t changed_properties_count = get_changed_properties_from_states(*current_animation_state, new_anim_state, changed_properties);

    // Keyframe changes state
    if (changed_properties.empty()) {
        on_close_inspector();
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
        auto& states = animations_data[get_animation_idx()].states;
        states.push_back(new_anim_state);
    }

    on_close_inspector();
}

void AnimationEditor::edit_keyframe(uint32_t index)
{
    show_keyframe_dirty = true;
    keyframe_dirty = true;
    editing_keyframe = true;

    auto& states = animations_data[get_animation_idx()].states;
    current_animation_state = &states[index];

    // Show submit_keyframe button
    auto w = Node2D::get_widget_from_name("submit_keyframe");
    w->set_visibility(true);

    // Hide create_keyframe button
    w = Node2D::get_widget_from_name("create_keyframe");
    w->set_visibility(false);

    // Deactivate open list
    w = Node2D::get_widget_from_name("open_list");
    static_cast<ui::Button2D*>(w)->set_disabled(true);
}

void AnimationEditor::duplicate_keyframe(uint32_t index)
{
    if (!current_node) {
        assert(0);
    }

    current_animation_state = get_animation_state(index);

    sAnimationState new_anim_state;
    new_anim_state.time = current_time;

    for (auto prop : current_animation_state->properties) {

        std::string property_name = prop.first;

        sPropertyState state = current_animation_state->properties[property_name];

        Track* track = current_animation->get_track_by_id(state.track_id);

        state.keyframe = &track->add_keyframe({ .value = state.value, .in = 0.0f, .out = 0.0f, .time = current_time });

        new_anim_state.properties[property_name] = state;
    }

    current_time += 0.5f;
    current_animation->recalculate_duration();
    auto& states = animations_data[get_animation_idx()].states;
    states.push_back(new_anim_state);
    current_animation_state = nullptr;
    keyframe_list_dirty = true;

    set_animation_state(states.size() - 1u);

    update_animation_trajectory();
}

sAnimationState* AnimationEditor::get_animation_state(uint32_t index)
{
    auto& states = animations_data[get_animation_idx()].states;
    return &states[index];
}

void AnimationEditor::set_animation_state(uint32_t index)
{
    Node3D* node = get_current_node();

    if (!node) {
        assert(0);
    }

    current_animation_state = get_animation_state(index);

    for (auto& p : current_animation_state->properties) {

        Node::AnimatableProperty node_property = node->get_animatable_property(p.first);
        void* data = node_property.property;

        current_animation->sample(current_animation_state->time, p.second.track_id, ANIMATION_LOOP_NONE, data);

        node->set_transform_dirty(true);

        if (dynamic_cast<Joint3D*>(node)) {
            current_node->set_transform_dirty(true);
        }
    }
}

void AnimationEditor::store_animation_state(sAnimationState& state)
{
    // Get properties always from the root node, since is the one
    // thas has all the properties!

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
        case Node::AnimatablePropertyType::INT8: state.properties[prop_it.first].value = *((int8_t*)data); break;
        case Node::AnimatablePropertyType::INT16: state.properties[prop_it.first].value = *((int16_t*)data); break;
        case Node::AnimatablePropertyType::INT32: state.properties[prop_it.first].value = *((int32_t*)data); break;
        case Node::AnimatablePropertyType::INT64: /*state.properties[prop_it.first].value = *((int64_t*)data);*/ break;
        case Node::AnimatablePropertyType::UINT8: state.properties[prop_it.first].value = *((uint8_t*)data); break;
        case Node::AnimatablePropertyType::UINT16: state.properties[prop_it.first].value = *((uint16_t*)data); break;
        case Node::AnimatablePropertyType::UINT32: state.properties[prop_it.first].value = *((uint32_t*)data); break;
        case Node::AnimatablePropertyType::UINT64: /*state.properties[prop_it.first].value = *((uint64_t*)data);*/ break;
        case Node::AnimatablePropertyType::FLOAT32: state.properties[prop_it.first].value = *((float*)data); break;
        case Node::AnimatablePropertyType::FLOAT64: /*state.properties[prop_it.first].value = *((long float*)data);*/ break;
        case Node::AnimatablePropertyType::IVEC2: state.properties[prop_it.first].value = *((glm::ivec2*)data); break;
        case Node::AnimatablePropertyType::UVEC2: state.properties[prop_it.first].value = *((glm::uvec2*)data); break;
        case Node::AnimatablePropertyType::FVEC2: state.properties[prop_it.first].value = *((glm::vec2*)data); break;
        case Node::AnimatablePropertyType::IVEC3: state.properties[prop_it.first].value = *((glm::ivec3*)data); break;
        case Node::AnimatablePropertyType::UVEC3: state.properties[prop_it.first].value = *((glm::uvec3*)data); break;
        case Node::AnimatablePropertyType::FVEC3: state.properties[prop_it.first].value = *((glm::vec3*)data); break;
        case Node::AnimatablePropertyType::IVEC4: state.properties[prop_it.first].value = *((glm::ivec4*)data); break;
        case Node::AnimatablePropertyType::UVEC4: state.properties[prop_it.first].value = *((glm::uvec4*)data); break;
        case Node::AnimatablePropertyType::FVEC4: state.properties[prop_it.first].value = *((glm::vec4*)data); break;
        case Node::AnimatablePropertyType::QUAT: state.properties[prop_it.first].value = *((glm::quat*)data); break;
        case Node::AnimatablePropertyType::UNDEFINED: break;
        }
    }
}

void AnimationEditor::set_loop_type(uint8_t type)
{
    if (!current_animation) {
        assert(0);
        return;
    }

    player->set_loop_type(type);
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
        w = Node2D::get_widget_from_name("pause_animation");
        w->set_visibility(true);
    }
}

void AnimationEditor::pause_animation()
{
    player->pause();

    {
        // Show play button
        auto w = Node2D::get_widget_from_name("play_animation");
        w->set_visibility(true);

        // Hide stop button
        w = Node2D::get_widget_from_name("pause_animation");
        w->set_visibility(false);
    }
}

void AnimationEditor::stop_animation()
{
    player->stop(true);

    {
        // Show play button
        auto w = Node2D::get_widget_from_name("play_animation");
        w->set_visibility(true);

        // Hide stop button
        w = Node2D::get_widget_from_name("pause_animation");
        w->set_visibility(false);
    }
}

void AnimationEditor::render_gui()
{
    ImGui::Text("Animations");

    ImGui::Separator();

    for (const auto& a : animations_data) {
        ImGui::Text("%s", a.second.animation->get_name().c_str());
    }

    ImGui::Separator();

    if (current_animation)
    {
        ImGui::Text("Animation %s", current_animation->get_name().c_str());
        ImGui::Text("Num Tracks %d", current_animation->get_track_count());

        auto& states = animations_data[get_animation_idx()].states;
        ImGui::Text("Num States %zu", states.size());

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

    main_panel = new ui::HContainer2D("animation_editor_root", { 48.0f, screen_size.y - 200.f }, ui::CREATE_3D);

    Node::bind("animation_editor_root@resize", (FuncUVec2)[&](const std::string& signal, glm::u32vec2 window_size) {
        main_panel->set_position({ 48.0f, window_size.y - 200.f });
    });

    ui::VContainer2D* vertical_container = new ui::VContainer2D("animation_vertical_container", { 0.0f, 0.0f });
    main_panel->add_child(vertical_container);

    // Add main row
    ui::HContainer2D* first_row = new ui::HContainer2D("row_0", { 0.0f, 0.0f });
    vertical_container->add_child(first_row);

    first_row->add_child(new ui::FloatSlider2D("animation_speed", "data/textures/animation_speed.png", player->get_speed()));

    // Animation settings
    {
        ui::ButtonSubmenu2D* loop_mode = new ui::ButtonSubmenu2D("loop_mode", { "data/textures/loop.png" });

        // ** Loop modes
        {
            ui::ComboButtons2D* combo_loops = new ui::ComboButtons2D("combo_loops");
            combo_loops->add_child(new ui::TextureButton2D("loop_none", { "data/textures/cross.png" }));
            combo_loops->add_child(new ui::TextureButton2D("loop_default", { "data/textures/loop.png", ui::SELECTED }));
            combo_loops->add_child(new ui::TextureButton2D("loop_reverse", { "data/textures/reverse_loop.png" }));
            combo_loops->add_child(new ui::TextureButton2D("loop_ping_pong", { "data/textures/ping_pong_loop.png" }));
            loop_mode->add_child(combo_loops);
        }

        first_row->add_child(loop_mode);
    }

    // ** Go back to scene editor **
    first_row->add_child(new ui::TextureButton2D("character_view", { "data/textures/character.png" }));

    ui::HContainer2D* second_row = new ui::HContainer2D("row_1", { 0.0f, 0.0f });
    vertical_container->add_child(second_row);

    // ** Go back to scene editor **
    second_row->add_child(new ui::TextureButton2D("go_back", { "data/textures/back.png" }));

    // ** Open keyframe list **
    second_row->add_child(new ui::TextureButton2D("open_list", { "data/textures/keyframe_list.png" }));

    // ** Keyframe actions
    {
        ui::ItemGroup2D* g_keyframes = new ui::ItemGroup2D("g_keyframes");
        g_keyframes->add_child(new ui::TextureButton2D("record_action", { "data/textures/record_action.png", ui::DISABLED }));
        g_keyframes->add_child(new ui::TextureButton2D("create_keyframe", { "data/textures/add_key.png" }));
        g_keyframes->add_child(new ui::TextureButton2D("submit_keyframe", { "data/textures/submit_key.png", ui::HIDDEN }));
        second_row->add_child(g_keyframes);
    }

    // ** Play animation **
    second_row->add_child(new ui::TextureButton2D("play_animation", { "data/textures/play.png" }));
    second_row->add_child(new ui::TextureButton2D("pause_animation", { "data/textures/pause.png", ui::HIDDEN }));
    second_row->add_child(new ui::TextureButton2D("stop_animation", { "data/textures/stop.png" }));

    // Create inspection panel (Nodes, properties, etc)
    inspector = new ui::Inspector({
        .name = "inspector_root",
        .title = "Animation",
        .position = {32.0f, 32.f},
        .close_fn = std::bind(&AnimationEditor::on_close_inspector, this, std::placeholders::_1),
        // .back_fn = std::bind(&AnimationEditor::on_goback_inspector, this, std::placeholders::_1)
    });

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
            left_hand_box->add_child(new ui::ImageLabel2D("Keyframe List", shortcuts::Y_BUTTON_PATH, shortcuts::OPEN_KEYFRAME_LIST));
            left_hand_box->add_child(new ui::ImageLabel2D("Play Animation", shortcuts::X_BUTTON_PATH, shortcuts::PLAY_ANIMATION));
        }

        // Right hand
        {
            right_hand_box = new ui::VContainer2D("right_controller_root", { 0.0f, 0.0f }, ui::CREATE_3D);
            right_hand_box->add_child(new ui::ImageLabel2D("Back to scene", shortcuts::B_BUTTON_PATH, shortcuts::BACK_TO_SCENE));
            right_hand_box->add_child(new ui::ImageLabel2D("Create keyframe", shortcuts::A_BUTTON_PATH, shortcuts::CREATE_KEYFRAME));
            right_hand_box->add_child(new ui::ImageLabel2D("Submit keyframe", shortcuts::A_BUTTON_PATH, shortcuts::SUBMIT_KEYFRAME));
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
        Node::bind("pause_animation", [&](const std::string& signal, void* button) { pause_animation(); });
        Node::bind("stop_animation", [&](const std::string& signal, void* button) { stop_animation(); });
    }

    // Loop mode events
    {
        Node::bind("loop_none", [&](const std::string& signal, void* button) { set_loop_type(ANIMATION_LOOP_NONE); });
        Node::bind("loop_default", [&](const std::string& signal, void* button) { set_loop_type(ANIMATION_LOOP_DEFAULT); });
        Node::bind("loop_reverse", [&](const std::string& signal, void* button) { set_loop_type(ANIMATION_LOOP_REVERSE); });
        Node::bind("loop_ping_pong", [&](const std::string& signal, void* button) { set_loop_type(ANIMATION_LOOP_PING_PONG); });
    }

    Node::bind("character_view", [&](const std::string& signal, void* button) {
        inspect_character(true);
    }); 

    Node::bind("animation_speed", (FuncFloat)[&](const std::string& signal, float value) { player->set_speed(value); });
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

    inspector->set_xr_transform(Transform::mat4_to_transform(m));

    inspector_transform_dirty = false;
}

void AnimationEditor::inspect_keyframe()
{
    inspector->clear(ui::INSPECTOR_FLAG_CLOSE_BUTTON, "Keyframe");

    inspect_keyframe_properties();

    inspect_node(get_current_node());

    Node::emit_signal(inspector->get_name() + "@children_changed", (void*)nullptr);

    show_keyframe_dirty = false;

    auto w = Node2D::get_widget_from_name("open_list");
    static_cast<ui::Button2D*>(w)->set_disabled(true);
}

void AnimationEditor::inspect_keyframe_properties()
{
    /*inspector->label("empty", "Keyframe Properties");
    inspector->same_line();
    std::string signal = node_name + std::to_string(node_signal_uid++) + "_intensity_slider";
    inspector->add_slider(signal, light->get_intensity(), 0.0f, 10.0f, 2);
    inspector->label("empty", "Interpolation", ui::SKIP_TEXT_RECT);
    inspector->end_line();*/
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

void AnimationEditor::inspect_character(bool force)
{
    if (!dynamic_cast<Character3D*>(current_node)) {
        assert(0);
        return;
    }

    inspector->clear();

    auto character = static_cast<Character3D*>(current_node);
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

    inspector_transform_dirty = !inspector->get_visibility() || force;

    if (force) {
        inspector->set_visibility(true);
    }
}

void AnimationEditor::inspect_keyframes_list(bool force)
{
    inspector->clear();

    auto& states = animations_data[get_animation_idx()].states;

    for (uint32_t i = 0; i < states.size(); ++i) {

        const sAnimationState& s = states[i];

        inspector->same_line();

        // add unique identifier for signals
        std::string key_name = "Keyframe" + std::to_string(i);

        // Edit keyframe
        {
            std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_edit";
            inspector->button(signal, "data/textures/edit.png");

            Node::bind(signal, [&, it = i](const std::string& sg, void* data) {
                edit_keyframe(it);
            });
        }

        // Duplicate keyframe
        {
            std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_duplicate";
            inspector->button(signal, "data/textures/duplicate_key.png");

            Node::bind(signal, [&, it = i](const std::string& sg, void* data) {
                duplicate_keyframe(it);
            });
        }

        // inspector->icon("data/textures/keyframe.png");
        // Set state
        {
            std::string signal = key_name + std::to_string(keyframe_signal_uid++) + "_label";
            inspector->label(signal, key_name);

            Node::bind(signal, [&, it = i](const std::string& sg, void* data) {
                set_animation_state(it);
            });
        }

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

    inspector_transform_dirty = !inspector->get_visibility() || force;

    if (force) {
        inspector->set_visibility(true);
    }
}

bool AnimationEditor::on_close_inspector(ui::Inspector* scope)
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
