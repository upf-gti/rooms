#include "rooms_engine.h"

#include "framework/nodes/environment_3d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/scene/parse_gltf.h"
#include "framework/utils/tinyfiledialogs.h"
#include "framework/utils/utils.h"
#include "framework/ui/io.h"
#include "framework/ui/keyboard.h"

#include "engine/scene.h"

#include "graphics/renderers/rooms_renderer.h"

#include "shaders/mesh_grid.wgsl.gen.h"
#include "shaders/ui/ui_ray_pointer.wgsl.gen.h"

#include "tools/sculpt/sculpt_editor.h"
#include "tools/scene/scene_editor.h"
#include "tools/tutorial_editor.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

#include <fstream>

bool RoomsEngine::use_environment_map = true;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
    int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    main_scene = new Scene();

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

    // Tutorial
    {
        tutorial_editor = new TutorialEditor();
        tutorial_editor->initialize();
    }

    // Set default editor..
    current_editor = scene_editor;

    // Grid
    {
        MeshInstance3D* grid_node = new MeshInstance3D();
        grid_node->set_name("Grid");
        grid_node->add_surface(RendererStorage::get_surface("quad"));
        grid_node->set_translation(glm::vec3(0.0f));
        grid_node->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        grid_node->scale(glm::vec3(10.f));

        Material grid_material;
        grid_material.priority = 100;
        grid_material.transparency_type = ALPHA_BLEND;
        grid_material.cull_type = CULL_NONE;
        grid_material.shader = RendererStorage::get_shader_from_source(shaders::mesh_grid::source, shaders::mesh_grid::path, grid_material);

        grid_node->set_surface_material_override(grid_node->get_surface(0), grid_material);

        main_scene->add_node(grid_node);
    }

    // Controller pointer
    {
        ray_pointer = parse_mesh("data/meshes/raycast.obj");

        Material pointer_material;
        pointer_material.transparency_type = ALPHA_BLEND;
        pointer_material.cull_type = CULL_NONE;
        pointer_material.shader = RendererStorage::get_shader_from_source(shaders::ui_ray_pointer::source, shaders::ui_ray_pointer::path, pointer_material);

        ray_pointer->set_surface_material_override(ray_pointer->get_surface(0), pointer_material);
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
        controller_mesh_right->set_model(Input::get_controller_pose(HAND_RIGHT));
        controller_mesh_left->set_model(Input::get_controller_pose(HAND_LEFT));
    }
}

void RoomsEngine::render()
{
    cursor.render();

    ui::Keyboard::render();

    if (use_environment_map) {
        environment->render();
    }

    main_scene->render();

    if (current_editor) {
        current_editor->render();
    }

    if (Renderer::instance->get_openxr_available()) {

        const glm::mat4x4& raycast_transform = Input::get_controller_pose(HAND_RIGHT, POSE_AIM);
        ray_pointer->set_model(raycast_transform);
        float xr_ray_distance = IO::get_xr_ray_distance();
        ray_pointer->scale(glm::vec3(1.0f, 1.0f, xr_ray_distance < 0.0f ? 0.5f : xr_ray_distance));

        ray_pointer->render();
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
}

void RoomsEngine::toggle_use_environment_map()
{
    use_environment_map = !use_environment_map;
}

bool RoomsEngine::export_scene()
{
    std::ofstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    // Write scene info
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    //const std::vector<Stroke>& strokes = rmr->getStrokeHistory();

    //file << "@" << strokes.size() << "\n";

    //glm::vec3 position = rmr->get_sculpt_start_position();
    //file << "@" << std::to_string(position.x) << " " + std::to_string(position.y) + " " + std::to_string(position.z) << "\n";

    //uint32_t num_edits = 0;

    //for (const Stroke& stroke : strokes)
    //{
    //    file << "@stroke " << stroke.stroke_id << " " << stroke.primitive << " " << stroke.operation << "\n";

    //    file << std::to_string(stroke.parameters.x) + " " + std::to_string(stroke.parameters.y) + " " + std::to_string(stroke.parameters.z) + " " + std::to_string(stroke.parameters.w) + "\n";

    //    file << "@stroke-material" << "\n";

    //    file << stroke.material.roughness << " " << stroke.material.metallic << " " << stroke.material.emissive << "\n";
    //    file << std::to_string(stroke.material.color.x) + " " + std::to_string(stroke.material.color.y) + " " + std::to_string(stroke.material.color.z) + " " + std::to_string(stroke.material.color.w) + "\n";
    //    file << std::to_string(stroke.material.noise_params.x) + " " + std::to_string(stroke.material.noise_params.y) + " " + std::to_string(stroke.material.noise_params.z) + " " + std::to_string(stroke.material.noise_params.w) + "\n";
    //    file << std::to_string(stroke.material.noise_color.x) + " " + std::to_string(stroke.material.noise_color.y) + " " + std::to_string(stroke.material.noise_color.z) + " " + std::to_string(stroke.material.noise_color.w) + "\n";

    //    // Add the edits for each stroke type

    //    file << "@stroke-edits" << " " << stroke.edit_count << "\n";

    //    for (size_t i = 0; i < stroke.edit_count; ++i)
    //    {
    //        const Edit& edit = stroke.edits[i];
    //        file << edit.to_string() << "\n";
    //        num_edits++;
    //    }
    //}

    //file.close();

    //spdlog::info("Scene exported! ({} edits)", num_edits);

    return true;
}

bool RoomsEngine::import_scene()
{
    std::ifstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    std::string line = "";

    //// Write scene info
    //RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    //RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    //// Num strokes
    //std::getline(file, line);
    //uint32_t num_strokes = std::stoi(line.substr(1));

    //uint32_t num_edits = 0;

    //// Starting sculpt position
    //std::getline(file, line);
    //glm::vec3 position = load_vec3(line.substr(1));
    //rmr->set_sculpt_start_position(position);
    //sculpt_editor->set_sculpt_started(true);

    //Stroke current_stroke;

    //// Parse edits
    //while (std::getline(file, line))
    //{
    //    std::string t = line.substr(1);
    //    auto tokens = tokenize(t);

    //    if (tokens[0] == "stroke")
    //    {
    //        // Push last stroke
    //        if (current_stroke.edit_count > 0) {
    //            rmr->push_stroke(current_stroke);
    //        }

    //        current_stroke.stroke_id = static_cast<uint32_t>(std::stoi(tokens[1]));
    //        current_stroke.primitive = static_cast<sdPrimitive>(std::stoi(tokens[2]));
    //        current_stroke.operation = static_cast<sdOperation>(std::stoi(tokens[3]));

    //        std::getline(file, line);
    //        current_stroke.parameters = load_vec4(line);
    //    }
    //    else if (tokens[0] == "stroke-material")
    //    {
    //        StrokeMaterial mat;

    //        // Roughness, metallic, emissive
    //        std::getline(file, line);
    //        glm::vec3 pbr_data = load_vec3(line);
    //        mat.roughness = pbr_data.x;
    //        mat.metallic = pbr_data.y;
    //        mat.emissive = pbr_data.z;

    //        // Color + noise parameters
    //        std::getline(file, line);
    //        mat.color = load_vec4(line);
    //        std::getline(file, line);
    //        mat.noise_params = load_vec4(line);
    //        std::getline(file, line);
    //        mat.noise_color = load_vec4(line);

    //        current_stroke.material = mat;
    //    }
    //    else if (tokens[0] == "stroke-edits")
    //    {
    //        uint32_t edit_count = std::stoi(tokens[1]);
    //        current_stroke.edit_count = edit_count;

    //        for (size_t i = 0; i < edit_count; ++i)
    //        {
    //            std::getline(file, line);
    //            current_stroke.edits[i].parse_string(line);
    //            num_edits++;
    //        }
    //    }
    //}

    //file.close();

    //// Push current (and last) stroke data
    //if (current_stroke.edit_count > 0) {
    //    rmr->push_stroke(current_stroke);
    //}

    ////rmr->compute_octree();

    //spdlog::info("Scene imported! ({} edits)", num_edits);

    return true;
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
