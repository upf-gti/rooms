#include "rooms_engine.h"

#include "framework/nodes/environment_3d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/sculpt_node.h"
#include "framework/nodes/skeleton_instance_3d.h"
#include "framework/nodes/player_node.h"
#include "framework/input.h"
#include "framework/parsers/parse_scene.h"
#include "framework/parsers/parse_gltf.h"
#include "framework/utils/utils.h"
#include "framework/utils/tinyfiledialogs.h"
#include "framework/ui/io.h"
#include "framework/ui/keyboard.h"
#include "framework/ui/context_menu.h"

#include "engine/scene.h"

#include "graphics/renderers/rooms_renderer.h"

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/mesh_grid.wgsl.gen.h"
#include "shaders/ui/ui_ray_pointer.wgsl.gen.h"

#include "tools/sculpt_editor.h"
#include "tools/scene_editor.h"
#include "tools/animation_editor.h"
#include "tools/player_editor.h"
#include "tools/tutorial_editor.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <fstream>
#include <graphics/managers/sculpt_manager.h>

bool RoomsEngine::use_grid = true;
bool RoomsEngine::use_environment_map = true;

int RoomsEngine::initialize(Renderer* renderer, sEngineConfiguration configuration)
{
    int error = Engine::initialize(renderer, configuration);

    if (error) return error;

	return 0;
}

int RoomsEngine::post_initialize()
{
    main_scene = new Scene("main_scene");

    environment = new Environment3D();

    // Load Meta Quest Controllers and Controller pointer
    if (renderer->get_openxr_available())
    {
        std::vector<Node*> entities;
        GltfParser parser;
        parser.parse("data/meshes/controllers/left_controller.glb", entities);
        parser.parse("data/meshes/controllers/right_controller.glb", entities);
        controller_mesh_left = static_cast<MeshInstance3D*>(entities[0]);
        controller_mesh_right = static_cast<MeshInstance3D*>(entities[1]);

        // Controller pointer
        ray_pointer = parse_mesh("data/meshes/raycast.obj");

        Material* pointer_material = new Material();
        pointer_material->set_transparency_type(ALPHA_BLEND);
        pointer_material->set_cull_type(CULL_NONE);
        pointer_material->set_type(MATERIAL_UNLIT);
        pointer_material->set_shader(RendererStorage::get_shader_from_source(shaders::ui_ray_pointer::source, shaders::ui_ray_pointer::path, shaders::ui_ray_pointer::libraries, pointer_material));

        ray_pointer->set_surface_material_override(ray_pointer->get_surface(0), pointer_material);

        sphere_pointer = parse_mesh("data/meshes/sphere.obj");

        Material* sphere_pointer_material = new Material();
        sphere_pointer_material->set_depth_read(false);
        sphere_pointer_material->set_priority(0);
        sphere_pointer_material->set_transparency_type(ALPHA_BLEND);
        sphere_pointer_material->set_type(MATERIAL_UNLIT);
        sphere_pointer_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries, sphere_pointer_material));

        sphere_pointer->set_surface_material_override(sphere_pointer->get_surface(0), sphere_pointer_material);
    }

    init_default_skeleton();

    // Instantiate and initialize editors
    {
        editors.resize(EDITOR_COUNT);

        editors[SCENE_EDITOR] = new SceneEditor("Scene Editor");
        editors[SCULPT_EDITOR] = new SculptEditor("Sculpt Editor");
        editors[ANIMATION_EDITOR] = new AnimationEditor("Animation Editor");
        editors[TUTORIAL_EDITOR] = new TutorialEditor("Tutorial Editor");
        editors[PLAYER_EDITOR] = new PlayerEditor("Player Editor");
    }

    for (auto editor : editors) {
        editor->initialize();
    }

    if (1 || skip_tutorial) {
        get_editor<TutorialEditor*>(TUTORIAL_EDITOR)->end();
    }

    // Set default editor..
    switch_editor(SCENE_EDITOR);

    // Grid
    {
        grid = new MeshInstance3D();
        grid->set_name("Grid");
        grid->add_surface(RendererStorage::get_surface("quad"));
        grid->set_position(glm::vec3(0.0f));
        grid->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        grid->scale(glm::vec3(25.f));

        grid->set_frustum_culling_enabled(false);

        Material* grid_material = new Material();
        grid_material->set_priority(100);
        grid_material->set_transparency_type(ALPHA_BLEND);
        grid_material->set_cull_type(CULL_NONE);
        grid_material->set_type(MATERIAL_UNLIT);
        grid_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_grid::source, shaders::mesh_grid::path, shaders::mesh_grid::libraries, grid_material));
        grid->set_surface_material_override(grid->get_surface(0), grid_material);
    }

    cursor.initialize();

    ui::Keyboard::initialize();

    {
        gizmo.initialize(TRANSLATE);

        // Gizmo events
        Node::bind("no_gizmo", [&](const std::string& signal, void* button) { gizmo.set_enabled(false); });
        Node::bind("translate", [&](const std::string& signal, void* button) { gizmo.set_operation(TRANSLATE); });
        Node::bind("rotate", [&](const std::string& signal, void* button) { gizmo.set_operation(ROTATE); });
        Node::bind("scale", [&](const std::string& signal, void* button) { gizmo.set_operation(SCALE); });
    }

    player = new PlayerNode();
    main_scene->add_node(player);

    return 0;
}

void RoomsEngine::clean()
{
    Engine::clean();

    for (auto editor : editors) {
        editor->clean();
    }

    gizmo.clean();
}

void RoomsEngine::set_main_scene(const std::string& scene_path)
{
    Engine::set_main_scene(scene_path);

    get_editor<SculptEditor*>(SCULPT_EDITOR)->set_current_sculpt(nullptr);
    get_editor<SceneEditor*>(SCENE_EDITOR)->set_main_scene(main_scene);
}

void RoomsEngine::add_to_main_scene(const std::string& scene_path)
{
    main_scene->parse(scene_path);
}

void RoomsEngine::init_default_skeleton()
{
    std::vector<Node*> nodes;
    GltfParser parser;
    parser.parse("data/meshes/character.glb", nodes);
    assert(nodes.size() == 1u);

    std::function<SkeletonInstance3D*(Node*)> find_skeleton = [&](Node* node)
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
        return (SkeletonInstance3D*)nullptr;
    };

    auto instance = find_skeleton(nodes[0]);

    default_skeleton = *instance->get_skeleton();
    default_skeleton.set_name("default_skeleton");

    delete nodes[0];
}

void RoomsEngine::update(float delta_time)
{
    bool is_xr = renderer->get_openxr_available();

    // Default cursor at the beginning of the frame..
    cursor.set(is_xr ? ui::MOUSE_CURSOR_CIRCLE : ui::MOUSE_CURSOR_DEFAULT);

    ui::Keyboard::update(delta_time);

    // NOTE: main update was before env_map update, test if this here breacks anything
    main_scene->update(delta_time);

    if (current_editor) {

        current_editor->update(delta_time);

        get_editor<TutorialEditor*>(TUTORIAL_EDITOR)->update(delta_time);
    }

    if (active_context_menu) {
        active_context_menu->update(delta_time);
    }

    cursor.update(delta_time);

    Node::check_controller_signals();

    if (use_environment_map) {
        environment->update(delta_time);
    }

    if (is_xr) {
        controller_mesh_right->set_transform(Transform::mat4_to_transform(Input::get_controller_pose(HAND_RIGHT)));
        controller_mesh_left->set_transform(Transform::mat4_to_transform(Input::get_controller_pose(HAND_LEFT)));
    }

    Engine::update(delta_time);
}

void RoomsEngine::render()
{
    if (show_imgui) {
        render_gui();
    }

    cursor.render();

    ui::Keyboard::render();

    if (use_environment_map) {
        environment->render();
    }

    bool playing_scene = dynamic_cast<PlayerEditor*>(current_editor);

    if (use_grid && !playing_scene) {
        grid->render();
    }

    main_scene->render();

    if (current_editor) {
        current_editor->render();
        get_editor<TutorialEditor*>(TUTORIAL_EDITOR)->render();
    }

    if(active_context_menu) {
        active_context_menu->render();
    }

    if (Renderer::instance->get_openxr_available() && !playing_scene) {
        const glm::mat4x4& raycast_transform = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        ray_pointer->set_transform(Transform::mat4_to_transform(raycast_transform));
        float xr_ray_distance = IO::get_xr_ray_distance();
        ray_pointer->scale(glm::vec3(1.0f, 1.0f, xr_ray_distance < 0.0f ? 0.5f : xr_ray_distance));
        ray_pointer->render();

        sphere_pointer->set_transform(Transform::mat4_to_transform(raycast_transform));
        sphere_pointer->scale(glm::vec3(0.1f));
        sphere_pointer->render();
    }

    Engine::render();

    // destroy pending stuff

    while (to_delete.size()) {
        Node* node = to_delete.back();
        delete node;
        to_delete.pop_back();
    }
}

void RoomsEngine::render_controllers()
{
    if (!Renderer::instance->get_openxr_available()) {
        return;
    }

    RoomsEngine* i = static_cast<RoomsEngine*>(instance);
    i->controller_mesh_right->render();
    i->controller_mesh_left->render();
}

void RoomsEngine::resize_window(int width, int height)
{
    Engine::resize_window(width, height);

    if (current_editor) {
        current_editor->on_resize_window(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }
}

void RoomsEngine::switch_editor(uint8_t editor_idx, void* data)
{
    if (editor_idx >= EDITOR_COUNT) {
        assert(0 && "Editor is not created!");
        return;
    }

    RoomsEngine* i = static_cast<RoomsEngine*>(instance);

    if (i->current_editor) {
        i->current_editor->on_exit();
    }

    i->current_editor = i->get_editor(editor_idx);
    i->current_editor_type = static_cast<EditorType>(editor_idx);
    i->current_editor->on_enter(data);
}

void RoomsEngine::toggle_use_grid()
{
    use_grid = !use_grid;
}

void RoomsEngine::toggle_use_environment_map()
{
    use_environment_map = !use_environment_map;
}

void RoomsEngine::set_current_sculpt(SculptNode* sculpt_instance)
{
    get_editor<SculptEditor*>(SCULPT_EDITOR)->set_current_sculpt(sculpt_instance);
}

void RoomsEngine::push_context_menu(ui::ContextMenu* cm)
{
    active_context_menu = cm;
}

void RoomsEngine::delete_context_menu(ui::ContextMenu* cm)
{
    active_context_menu = nullptr;

    IO::blur();

    to_delete.push_back(cm);
}

void RoomsEngine::render_gui()
{
    render_default_gui();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open room (.room)"))
            {
                std::vector<const char*> filter_patterns = { "*.room" };
                char const* open_file_name = tinyfd_openFileDialog(
                    "Room loader",
                    "",
                    filter_patterns.size(),
                    filter_patterns.data(),
                    "Rooms format",
                    0
                );

                if (open_file_name) {
                    set_main_scene(open_file_name);
                }
            }
            if (ImGui::MenuItem("Save room (.room)"))
            {
                std::vector<const char*> filter_patterns = { "*.room" };

                char const* save_file_name = tinyfd_saveFileDialog(
                    "Room loader",
                    "",
                    filter_patterns.size(),
                    filter_patterns.data(),
                    "Rooms format"
                );

                if (save_file_name) {
                    main_scene->serialize(save_file_name);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

    bool active = true;
    ImGui::Begin("Debug panel", &active, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoFocusOnAppearing);

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("TabBar", tab_bar_flags))
    {
        if (current_editor && ImGui::BeginTabItem(current_editor->get_name().c_str()))
        {
            current_editor->render_gui();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rooms Debugger"))
        {
            sGPU_SculptResults &intersection_info = rooms_renderer->get_sculpt_manager()->loaded_results;
            std::string intersected = (intersection_info.ray_intersection.has_intersected == 1u) ? "yes" : "no";
            ImGui::Text("Ray Intersection: %s", intersected.c_str());
            ImGui::Text("Tile pointer: %d", intersection_info.ray_intersection.tile_pointer);
            ImGui::ColorEdit3("Picked albedo:", (float*)&intersection_info.ray_intersection.intersection_albedo);
            ImGui::Text("Picked metallic: %.3f", intersection_info.ray_intersection.intersection_metallic);
            ImGui::Text("Picked roughness: %.3f", intersection_info.ray_intersection.intersection_roughness);
            if (intersection_info.ray_intersection.has_intersected) {
                ImGui::Text("Intersection t : %.3f", intersection_info.ray_intersection.ray_t);
            }

            ImGui::Separator();

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(0, 255, 0, 255)));
            ImGui::Text("Timestamp queries:");
            ImGui::PopStyleColor();

            std::vector<float> timestamps = renderer->get_last_frame_timestamps();
            std::map<uint8_t, std::string>& queries_map = renderer->get_queries_label_map();

            ImGui::Text("\tlast evaluation time: %.4f", rooms_renderer->get_last_evaluation_time());

            for (int i = 0; i < timestamps.size(); ++i) {
                float time = timestamps[i];
                std::string label = queries_map[i * 2 + 1];

                ImGui::Text(("\t" + label + ": %.4f").c_str(), time);
            }

            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}


