#include "rooms_engine.h"

#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/scene/parse_gltf.h"
#include "graphics/renderers/rooms_renderer.h"

#include "spdlog/spdlog.h"

#include "backends/imgui_impl_wgpu.h"
#include "backends/imgui_impl_glfw.h"

#include "framework/utils/tinyfiledialogs.h"

#include <fstream>

EntityMesh* RoomsEngine::skybox = nullptr;
std::vector<Entity*> RoomsEngine::entities;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    sculpt_editor.initialize();

    skybox = parse_mesh("data/meshes/cube.obj");
    skybox->set_surface_material_shader(0, RendererStorage::get_shader("data/shaders/mesh_texture_cube.wgsl"));
    skybox->set_surface_material_diffuse(0, Renderer::instance->get_irradiance_texture());
    skybox->scale(glm::vec3(100.f));
    skybox->set_surface_material_priority(0, 2);

    //if (parse_scene("data/gltf_tests/Sponza/Sponza.gltf", entities)) {
    //    //Renderer::instance->get_camera()->look_at_entity(entities.back());
    //}

    // import_scene();

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    sculpt_editor.clean();
}

void RoomsEngine::update(float delta_time)
{
    Engine::update(delta_time);

    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    skybox->set_translation(renderer->get_camera_eye());

    sculpt_editor.update(delta_time);

    if (Input::was_key_pressed(GLFW_KEY_E))
    {
        export_scene();
    }
}

void RoomsEngine::render()
{

#ifndef __EMSCRIPTEN__
    render_gui();
#endif

    skybox->render();

	for (auto entity : entities) {
		entity->render();
	}

    sculpt_editor.render();

	Engine::render();
}

bool RoomsEngine::export_scene()
{
    std::ofstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    // Write scene info
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    const std::vector<Stroke>& strokes = rmr->getStrokeHistory();

    file << "@" << strokes.size() << "\n";

    glm::vec3 position = rmr->get_sculpt_start_position();
    file << "@" << std::to_string(position.x) << " " + std::to_string(position.y) + " " + std::to_string(position.z) << "\n";

    uint32_t num_edits = 0;

    for (const Stroke& stroke : strokes)
    {
        file << "@stroke " << stroke.stroke_id << "\n";

        file << stroke.primitive << "\n";
        file << stroke.operation << "\n";
        file << std::to_string(stroke.parameters.x) + " " + std::to_string(stroke.parameters.y) + " " + std::to_string(stroke.parameters.z) + " " + std::to_string(stroke.parameters.w) + "\n";

        file << "@stroke-material" << "\n";

        file << stroke.material.roughness << " " << stroke.material.metallic << " " << stroke.material.emissive << "\n";
        file << std::to_string(stroke.material.color.x) + " " + std::to_string(stroke.material.color.y) + " " + std::to_string(stroke.material.color.z) + " " + std::to_string(stroke.material.color.w) + "\n";
        file << std::to_string(stroke.material.noise_params.x) + " " + std::to_string(stroke.material.noise_params.y) + " " + std::to_string(stroke.material.noise_params.z) + " " + std::to_string(stroke.material.noise_params.w) + "\n";
        file << std::to_string(stroke.material.noise_color.x) + " " + std::to_string(stroke.material.noise_color.y) + " " + std::to_string(stroke.material.noise_color.z) + " " + std::to_string(stroke.material.noise_color.w) + "\n";

        // Add the edits for each stroke type

        file << "@stroke-edits" << " " << stroke.edit_count << "\n";

        for (size_t i = 0; i < stroke.edit_count; ++i)
        {
            const Edit& edit = stroke.edits[i];
            file << edit.to_string() << "\n";
            num_edits++;
        }
    }

    file.close();

    spdlog::info("Scene exported! ({} edits)", num_edits);

    return true;
}

bool RoomsEngine::import_scene()
{
    std::ifstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    std::string line = "";

    // Write scene info
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    uint32_t num_strokes = 0;
    uint32_t num_edits = 0;

    int stroke_it = -1;

    // Num strokes
    std::getline(file, line);
    num_strokes = std::stoi(line.substr(1));

    // Starting sculpt position
    std::getline(file, line);
    glm::vec3 position = load_vec3(line.substr(1));
    rmr->set_sculpt_start_position(position);
    sculpt_editor.set_sculpt_started(true);

    std::vector<Stroke> strokes;
    strokes.resize(num_strokes);

    // Parse edits
    while (std::getline(file, line))
    {
        std::string t = line.substr(1);
        auto tokens = tokenize(t);

        if (tokens[0] == "stroke")
        {
            stroke_it++;
            strokes[stroke_it].stroke_id = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "stroke-material")
        {
            // Roughness, metallic, emissive
            std::getline(file, line);
            auto lt = tokenize(line);
            strokes[stroke_it].material.roughness = std::stof(lt[0]);
            strokes[stroke_it].material.metallic = std::stof(lt[1]);
            strokes[stroke_it].material.emissive = std::stof(lt[2]);

            // Color + noise parameters
            std::getline(file, line);
            strokes[stroke_it].material.color = load_vec4(line);
            std::getline(file, line);
            strokes[stroke_it].material.noise_params = load_vec4(line);
            std::getline(file, line);
            strokes[stroke_it].material.noise_color = load_vec4(line);
        }
        else if (tokens[0] == "stroke-edits")
        {
            uint32_t edit_count = std::stoi(tokens[1]);
            strokes[stroke_it].edit_count = edit_count;

            for (size_t i = 0; i < edit_count; ++i)
            {
                std::getline(file, line);
                strokes[stroke_it].edits[i].parse_string(line);
                num_edits++;
            }
        }
    }

    file.close();

    /*if (edit_count != scene_header.num_edits)
    {
        spdlog::error("[import_scene] Some edits couldn't be imported!");
        return false;
    }*/

    // Merge them into the scene in chunks of 64

    /*int chunk_size = 64;
    int chunks = ceil((float)edit_count / chunk_size);

    for (int i = 0; i < chunks; ++i)
    {
        int start_index = i * chunk_size;
        int end_index = std::min(start_index + chunk_size, scene_header.num_edits);

        for (int j = start_index; j < end_index; ++j)
        {
            rmr->push_edit( edits[j] );
            edit_count--;
        }

        rmr->compute_octree();
    }

    spdlog::info("Scene imported! ({} edits, {} left)", scene_header.num_edits, edit_count);*/
    
    return true;
}

void RoomsEngine::render_gui()
{
    if (RoomsRenderer::instance->get_openxr_available()) {
        return;
    }
    bool active = true;

    ImGui::SetNextWindowSize({ 300, 400 });
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
                    parse_scene(open_file_name, entities);
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
                        entities.clear();
                        ImGui::CloseCurrentPopup();
                    }

                    ImGui::EndPopup();
                }

                std::vector<Entity*>::iterator it = entities.begin();
                while (it != entities.end())
                {
                    if (show_tree_recursive(*it)) {
                        it = entities.erase(it);
                    }
                    else {
                        it++;
                    }
                }
                
                ImGui::TreePop();
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Sculpt Editor"))
        {
            sculpt_editor.render_gui();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Debugger"))
        {
            const RayIntersectionInfo& info = static_cast<RoomsRenderer*>(RoomsRenderer::instance)->get_raymarching_renderer()->get_ray_intersection_info();
            std::string intersected = info.intersected ? "yes" : "no";
            ImGui::Text(("Ray Intersection: " + intersected).c_str());
            ImGui::Text(("Tile pointer: " + std::to_string(info.tile_pointer)).c_str());
            ImGui::ColorEdit3("Picked albedo:", (float*) &info.material_albedo);
            if (info.intersected) {
                ImGui::Text("Intersection position :");
                ImGui::Text("   : %.3f, %.3f, %.3f", info.intersection_position.x, info.intersection_position.y, info.intersection_position.z);
            }

            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::Separator();

    ImGui::End();
}

bool RoomsEngine::show_tree_recursive(Entity* entity)
{
    std::vector<Entity*>& children = entity->get_children();

    EntityMesh* entity_mesh = dynamic_cast<EntityMesh*>(entity);

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;

    if (!entity_mesh && children.empty() || (entity_mesh && children.empty() && entity_mesh->get_surfaces().empty())) {
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

        std::vector<Entity*>::iterator it = children.begin();

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
