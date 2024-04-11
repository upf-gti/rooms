#include "rooms_engine.h"
#include "framework/nodes/environment_3d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/nodes/omni_light_3d.h"
#include "framework/nodes/spot_light_3d.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/scene/parse_gltf.h"
#include "graphics/renderers/rooms_renderer.h"
#include "tools/sculpt/sculpt_editor.h"

#include "spdlog/spdlog.h"
#include "imgui.h"
#include "framework/utils/tinyfiledialogs.h"
#include "framework/utils/ImGuizmo.h"

#include <fstream>

std::vector<Node3D*> RoomsEngine::entities;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
    int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    sculpt_editor = new SculptEditor;

    sculpt_editor->initialize();

    skybox = new Environment3D();

    entities.push_back(skybox);

    //if (parse_scene("data/gltf_tests/Sponza/Sponza.gltf", entities)) {
    //    //Renderer::instance->get_camera()->look_at_entity(entities.back());
    //}

    //import_scene();

    /*OmniLight3D* omni_light = new OmniLight3D();
    omni_light->set_name("omni_light");
    omni_light->set_translation({ 1.0f, 1.f, 0.0f });
    omni_light->set_color({ 1.0f, 1.0f, 1.0f });
    omni_light->set_intensity(1.0f);
    omni_light->set_range(5.0f);

    entities.push_back(omni_light);
    RoomsRenderer::instance->add_light(omni_light);
    */

    SpotLight3D* spot_light = new SpotLight3D();
    spot_light->set_name("spot_light");
    spot_light->set_translation({ 0.0f, 1.f, 0.0f });
    spot_light->rotate(glm::radians(-90.f), { 1.f, 0.0f, 0.f });
    spot_light->set_color({ 1.0f, 1.0f, 1.0f });
    spot_light->set_intensity(1.0f);
    spot_light->set_range(5.0f);

    entities.push_back(spot_light);
    RoomsRenderer::instance->add_light(spot_light);

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    Node2D::clean();

    sculpt_editor->clean();
}

void RoomsEngine::update(float delta_time)
{
    Engine::update(delta_time);

    Node::check_controller_signals();

    for (auto entity : entities) {
        entity->update(delta_time);
    }

    sculpt_editor->update(delta_time);

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
    
	for (auto entity : entities) {
		entity->render();
	}

    sculpt_editor->render();

    {
        static glm::mat4x4 test_model = glm::mat4x4(1.0f);
        Camera* camera = RoomsRenderer::instance->get_camera();
        gizmo.render(camera->get_view(), camera->get_projection(), test_model);
    }

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
        file << "@stroke " << stroke.stroke_id << " " << stroke.primitive << " " << stroke.operation << "\n";

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

    // Num strokes
    std::getline(file, line);
    uint32_t num_strokes = std::stoi(line.substr(1));

    uint32_t num_edits = 0;

    // Starting sculpt position
    std::getline(file, line);
    glm::vec3 position = load_vec3(line.substr(1));
    rmr->set_sculpt_start_position(position);
    sculpt_editor->set_sculpt_started(true);

    Stroke current_stroke;

    // Parse edits
    while (std::getline(file, line))
    {
        std::string t = line.substr(1);
        auto tokens = tokenize(t);

        if (tokens[0] == "stroke")
        {
            // Push last stroke
            if (current_stroke.edit_count > 0) {
                rmr->push_stroke(current_stroke);
            }

            current_stroke.stroke_id = static_cast<uint32_t>(std::stoi(tokens[1]));
            current_stroke.primitive = static_cast<sdPrimitive>(std::stoi(tokens[2]));
            current_stroke.operation = static_cast<sdOperation>(std::stoi(tokens[3]));

            std::getline(file, line);
            current_stroke.parameters = load_vec4(line);
        }
        else if (tokens[0] == "stroke-material")
        {
            StrokeMaterial mat;

            // Roughness, metallic, emissive
            std::getline(file, line);
            glm::vec3 pbr_data = load_vec3(line);
            mat.roughness = pbr_data.x;
            mat.metallic = pbr_data.y;
            mat.emissive = pbr_data.z;

            // Color + noise parameters
            std::getline(file, line);
            mat.color = load_vec4(line);
            std::getline(file, line);
            mat.noise_params = load_vec4(line);
            std::getline(file, line);
            mat.noise_color = load_vec4(line);

            current_stroke.material = mat;
        }
        else if (tokens[0] == "stroke-edits")
        {
            uint32_t edit_count = std::stoi(tokens[1]);
            current_stroke.edit_count = edit_count;

            for (size_t i = 0; i < edit_count; ++i)
            {
                std::getline(file, line);
                current_stroke.edits[i].parse_string(line);
                num_edits++;
            }
        }
    }

    file.close();

    // Push current (and last) stroke data
    if (current_stroke.edit_count > 0) {
        rmr->push_stroke(current_stroke);
    }

    rmr->compute_octree();

    spdlog::info("Scene imported! ({} edits)", num_edits);
    
    return true;
}

void RoomsEngine::render_gui()
{
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

                std::vector<Node3D*>::iterator it = entities.begin();
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
            sculpt_editor->render_gui();
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

            bool msaa_enabled = Renderer::instance->get_msaa_count() != 1;

            if (ImGui::Checkbox("Enable MSAAx4", &msaa_enabled)) {
                if (msaa_enabled) {
                    Renderer::instance->set_msaa_count(4);
                }
                else {
                    Renderer::instance->set_msaa_count(1);
                }
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
