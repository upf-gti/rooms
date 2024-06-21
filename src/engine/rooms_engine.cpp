#include "rooms_engine.h"

#include "framework/nodes/custom_node_factory.h"
#include "framework/nodes/environment_3d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/sculpt_instance.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/scene/parse_gltf.h"
#include "framework/utils/tinyfiledialogs.h"
#include "framework/utils/utils.h"
#include "framework/ui/io.h"
#include "framework/ui/keyboard.h"

#include "engine/scene.h"

#include "graphics/renderers/rooms_renderer.h"

#include "shaders/mesh_color.wgsl.gen.h"
#include "shaders/mesh_grid.wgsl.gen.h"
#include "shaders/ui/ui_ray_pointer.wgsl.gen.h"

#include "tools/sculpt/sculpt_editor.h"
#include "tools/scene/scene_editor.h"
#include "tools/tutorial_editor.h"
#include "framework/nodes/sculpt_instance.h"


#include "spdlog/spdlog.h"
#include "imgui.h"

#include <fstream>

bool RoomsEngine::use_grid = true;
bool RoomsEngine::use_environment_map = true;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
    int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    node_factory = custom_node_factory;

    main_scene = new Scene("main_scene");

    environment = new Environment3D();

    // Meta Quest Controllers
    if (renderer->get_openxr_available())
    {
        std::vector<Node*> entities;
        parse_gltf("data/meshes/controllers/left_controller.glb", entities);
        parse_gltf("data/meshes/controllers/right_controller.glb", entities);
        controller_mesh_left = static_cast<MeshInstance3D*>(entities[0]);
        controller_mesh_right = static_cast<MeshInstance3D*>(entities[1]);
    }

    // Tutorial
    {
        tutorial_editor = new TutorialEditor();
        tutorial_editor->initialize();
    }

    // Scenes
    {
        scene_editor = new SceneEditor();
        scene_editor->initialize();
    }

    // Sculpting
    {
        sculpt_editor = new SculptEditor();
        sculpt_editor->initialize();
    }

    // Set default editor..
    switch_editor(EditorType::SCENE_EDITOR);

    // Grid
    {
        grid = new MeshInstance3D();
        grid->set_name("Grid");
        grid->add_surface(RendererStorage::get_surface("quad"));
        grid->set_position(glm::vec3(0.0f));
        grid->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        grid->scale(glm::vec3(10.f));

        Material grid_material;
        grid_material.priority = 100;
        grid_material.transparency_type = ALPHA_BLEND;
        grid_material.cull_type = CULL_NONE;
        grid_material.shader = RendererStorage::get_shader_from_source(shaders::mesh_grid::source, shaders::mesh_grid::path, grid_material);
        grid->set_surface_material_override(grid->get_surface(0), grid_material);
    }

    // Controller pointer
    {
        ray_pointer = parse_mesh("data/meshes/raycast.obj");

        Material pointer_material;
        pointer_material.transparency_type = ALPHA_BLEND;
        pointer_material.cull_type = CULL_NONE;
        pointer_material.shader = RendererStorage::get_shader_from_source(shaders::ui_ray_pointer::source, shaders::ui_ray_pointer::path, pointer_material);

        ray_pointer->set_surface_material_override(ray_pointer->get_surface(0), pointer_material);

        sphere_pointer = parse_mesh("data/meshes/sphere.obj");

        pointer_material = {};
        pointer_material.shader = RendererStorage::get_shader_from_source(shaders::mesh_color::source, shaders::mesh_color::path, pointer_material);

        sphere_pointer->set_surface_material_override(sphere_pointer->get_surface(0), pointer_material);
    }

    cursor.load();

    ui::Keyboard::initialize();

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    Node2D::clean();

    if (scene_editor) {
        scene_editor->clean();
    }

    if (sculpt_editor) {
        sculpt_editor->clean();
    }
}

void RoomsEngine::set_main_scene(const std::string& scene_path)
{
    delete main_scene;

    sculpt_editor->set_current_sculpt(nullptr);

    main_scene = new Scene();
    main_scene->parse(scene_path);

    scene_editor->set_main_scene(main_scene);
}

void RoomsEngine::add_to_main_scene(const std::string& scene_path)
{
    main_scene->parse(scene_path);
}

void RoomsEngine::update(float delta_time)
{
    bool is_xr = renderer->get_openxr_available();

    // Default cursor at the beginning of the frame..
    cursor.set(is_xr ? ui::MOUSE_CURSOR_CIRCLE : ui::MOUSE_CURSOR_DEFAULT);

    ui::Keyboard::update(delta_time);

    if (current_editor) {
        current_editor->update(delta_time);
    }

    cursor.update(delta_time);

    Node::check_controller_signals();

    Engine::update(delta_time);

    if (use_environment_map) {
        environment->update(delta_time);
    }

    main_scene->update(delta_time);

    if (is_xr) {
        controller_mesh_right->set_transform(Transform::mat4_to_transform(Input::get_controller_pose(HAND_RIGHT)));
        controller_mesh_left->set_transform(Transform::mat4_to_transform(Input::get_controller_pose(HAND_LEFT)));
    }
}

void RoomsEngine::render()
{
    cursor.render();

    ui::Keyboard::render();

    if (use_environment_map) {
        environment->render();
    }

    if (use_grid) {
        grid->render();
    }

    main_scene->render();

    if (current_editor) {
        current_editor->render();
    }

    if (Renderer::instance->get_openxr_available()) {

        const glm::mat4x4& raycast_transform = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        ray_pointer->set_transform(Transform::mat4_to_transform(raycast_transform));
        float xr_ray_distance = IO::get_xr_ray_distance();
        ray_pointer->scale(glm::vec3(1.0f, 1.0f, xr_ray_distance < 0.0f ? 0.5f : xr_ray_distance));
        ray_pointer->render();

        sphere_pointer->set_transform(Transform::mat4_to_transform(raycast_transform));
        sphere_pointer->scale(glm::vec3(0.1f));
        sphere_pointer->render();
    }

#ifndef __EMSCRIPTEN__
    render_gui();
#endif

    Engine::render();
}

void RoomsEngine::render_controllers()
{
    RoomsEngine* i = static_cast<RoomsEngine*>(instance);

    if (i->renderer->get_openxr_available()) {
        i->controller_mesh_right->render();
        i->controller_mesh_left->render();
    }
}

void RoomsEngine::switch_editor(uint8_t editor)
{
    RoomsEngine* i = static_cast<RoomsEngine*>(instance);

    switch (editor)
    {
    case TUTORIAL_EDITOR:
        i->current_editor = i->tutorial_editor;
        break;
    case SCENE_EDITOR:
        i->current_editor = i->scene_editor;
        break;
    case SCULPT_EDITOR:
        i->current_editor = i->sculpt_editor;
        break;
    case SHAPE_EDITOR:
        // ...
        break;
    default:
        assert(0 && "Editor is not created!");
        break;
    }

    i->current_editor_type = (EditorType) editor;
}

void RoomsEngine::toggle_use_grid()
{
    use_grid = !use_grid;
}

void RoomsEngine::toggle_use_environment_map()
{
    use_environment_map = !use_environment_map;
}

void RoomsEngine::set_current_sculpt(SculptInstance* sculpt_instance)
{
    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    rooms_renderer->get_raymarching_renderer()->set_current_sculpt(sculpt_instance);

    sculpt_editor->set_current_sculpt(sculpt_instance);
}

void RoomsEngine::render_gui()
{
    bool active = true;

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);

    // ImGui::SetNextWindowSize({ 300, 400 });
    ImGui::Begin("Debug panel", &active, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoFocusOnAppearing);

    if (ImGui::BeginMenuBar())
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
            if (ImGui::MenuItem("Open scene (.gltf, .glb, .obj)"))
            {
                std::vector<const char*> filter_patterns = { "*.gltf", "*.glb", "*.obj" };
                char const* open_file_name = tinyfd_openFileDialog(
                    "Scene loader",
                    "",
                    filter_patterns.size(),
                    filter_patterns.data(),
                    "Scene formats",
                    0
                );

                if (open_file_name) {
                    std::vector<Node*> entities;
                    parse_scene(open_file_name, entities);
                    main_scene->add_nodes(entities);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("TabBar", tab_bar_flags))
    {
        if (ImGui::BeginTabItem("Scene"))
        {
            if (ImGui::TreeNodeEx("Root", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::BeginPopupContextItem()) // <-- use last item id as popup id
                {
                    if (ImGui::Button("Delete All")) {
                        main_scene->delete_all();
                        ImGui::CloseCurrentPopup();
                    }

                    ImGui::EndPopup();
                }

                std::vector<Node*>& nodes = main_scene->get_nodes();
                std::vector<Node*>::iterator it = nodes.begin();
                while (it != nodes.end())
                {
                    if (show_tree_recursive(*it)) {
                        it = nodes.erase(it);
                    }
                    else {
                        it++;
                    }
                }

                ImGui::TreePop();
            }
            ImGui::EndTabItem();
        }
        if (scene_editor && ImGui::BeginTabItem("Scene Editor"))
        {
            scene_editor->render_gui();
            ImGui::EndTabItem();
        }
        if (sculpt_editor && ImGui::BeginTabItem("Sculpt Editor"))
        {
            sculpt_editor->render_gui();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Debugger"))
        {
            const RayIntersectionInfo& info = rooms_renderer->get_raymarching_renderer()->get_ray_intersection_info();
            std::string intersected = info.intersected ? "yes" : "no";
            ImGui::Text(("Ray Intersection: " + intersected).c_str());
            ImGui::Text(("Tile pointer: " + std::to_string(info.tile_pointer)).c_str());
            ImGui::ColorEdit3("Picked albedo:", (float*) &info.material_albedo);
            if (info.intersected) {
                ImGui::Text("Intersection position :");
                ImGui::Text("   : %.3f, %.3f, %.3f", info.intersection_position.x, info.intersection_position.y, info.intersection_position.z);
            }

            bool msaa_enabled = Renderer::instance->get_msaa_count() != 1;

            if (ImGui::Checkbox("Enable MSAAx4", &msaa_enabled)) {
                if (msaa_enabled) {
                    Renderer::instance->set_msaa_count(4);
                }
                else {
                    Renderer::instance->set_msaa_count(1);
                }
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
    ImGui::Separator();

    ImGui::End();
}

bool RoomsEngine::show_tree_recursive(Node* entity)
{
    std::vector<Node*>& children = entity->get_children();

    MeshInstance3D* entity_mesh = dynamic_cast<MeshInstance3D*>(entity);

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;

    if ((entity_mesh && children.empty() && entity_mesh->get_surfaces().empty())) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }

    if (ImGui::TreeNodeEx(entity->get_name().c_str(), flags))
    {
        if (ImGui::BeginPopupContextItem()) // <-- use last item id as popup id
        {
            if (ImGui::Button("Delete")) {
                ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
                ImGui::TreePop();

                if (static_cast<SculptInstance*>(entity) != nullptr) {
                    dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->remove_sculpt_instance((SculptInstance*) entity);
                }
                return true;
            }
            ImGui::EndPopup();
        }

        if (entity_mesh) {

            const std::vector<Surface*>& surfaces = entity_mesh->get_surfaces();

            for (int i = 0; i < surfaces.size(); ++i) {

                ImGui::TreeNodeEx(("Surface " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Leaf);
                ImGui::TreePop();
            }
        }

        entity->render_gui();

        std::vector<Node*>::iterator it = children.begin();

        while (it != children.end())
        {
            if (show_tree_recursive(*it)) {
                it = children.erase(it);
            }
            else {
                it++;
            }
        }

        ImGui::TreePop();
    }

    return false;
}
