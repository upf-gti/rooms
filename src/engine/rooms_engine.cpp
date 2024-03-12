#include "rooms_engine.h"
#include "framework/nodes/environment_3d.h"
#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "framework/scene/parse_gltf.h"
#include "graphics/renderers/rooms_renderer.h"

#include "spdlog/spdlog.h"
#include "imgui.h"
#include "framework/utils/tinyfiledialogs.h"

#include <fstream>

std::vector<Node3D*> RoomsEngine::entities;

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
    int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    sculpt_editor.initialize();

    skybox = new Environment3D();

    entities.push_back(skybox);

    //if (parse_scene("data/gltf_tests/Sponza/Sponza.gltf", entities)) {
    //    //Renderer::instance->get_camera()->look_at_entity(entities.back());
    //}

    //import_scene();

    root_2d = new ui::HContainer2D("root", { 12.0f, 12.f });

    {
        ui::ItemGroup2D* g_main_tools = new ui::ItemGroup2D("g_main_tools");
        g_main_tools->add_child(new ui::TextureButton2D("sculpt", "data/textures/cube.png", ui::UNIQUE_SELECTION | ui::SELECTED));
        g_main_tools->add_child(new ui::TextureButton2D("paint", "data/textures/paint.png", ui::UNIQUE_SELECTION));
        root_2d->add_child(g_main_tools);
    }

    {
        ui::ButtonSubmenu2D* primitives_submenu = new ui::ButtonSubmenu2D("primitives", "data/textures/primitives.png");

        {
            ui::ItemGroup2D* g0_primitives = new ui::ItemGroup2D("g0_primitives");
            g0_primitives->add_child(new ui::TextureButton2D("sphere", "data/textures/sphere.png", ui::UNIQUE_SELECTION | ui::SELECTED));
            g0_primitives->add_child(new ui::TextureButton2D("cube", "data/textures/cube.png", ui::UNIQUE_SELECTION));
            g0_primitives->add_child(new ui::TextureButton2D("cone", "data/textures/cone.png", ui::UNIQUE_SELECTION));
            g0_primitives->add_child(new ui::TextureButton2D("capsule", "data/textures/capsule.png", ui::UNIQUE_SELECTION));
            g0_primitives->add_child(new ui::TextureButton2D("cylinder", "data/textures/cylinder.png", ui::UNIQUE_SELECTION));
            g0_primitives->add_child(new ui::TextureButton2D("torus", "data/textures/torus.png", ui::UNIQUE_SELECTION));
            g0_primitives->add_child(new ui::TextureButton2D("bezier", "data/textures/bezier.png", ui::UNIQUE_SELECTION));
            primitives_submenu->add_child(g0_primitives);
        }

        {
            ui::ButtonSubmenu2D* shape_editor_submenu = new ui::ButtonSubmenu2D("shape_editor", "data/textures/shape_editor.png");

            {
                ui::ItemGroup2D* g_onion = new ui::ItemGroup2D("g_onion");
                g_onion->add_child(new ui::TextureButton2D("onion", "data/textures/onion.png", ui::UNIQUE_SELECTION | ui::ALLOW_TOGGLE));
                g_onion->add_child(new ui::Slider2D("onion_value", 0.0f, ui::SliderMode::VERTICAL, 0.01f));
                shape_editor_submenu->add_child(g_onion);
            }

            {
                ui::ItemGroup2D* g_capped = new ui::ItemGroup2D("g_capped");
                g_capped->add_child(new ui::TextureButton2D("capped", "data/textures/capped.png", ui::UNIQUE_SELECTION | ui::ALLOW_TOGGLE));
                g_capped->add_child(new ui::Slider2D("cap_value", 0.0f, ui::SliderMode::VERTICAL, 0.01f));
                shape_editor_submenu->add_child(g_capped);
            }

            primitives_submenu->add_child(shape_editor_submenu);
        }

        root_2d->add_child(primitives_submenu);
    }

    {
        ui::ButtonSubmenu2D* material_submenu = new ui::ButtonSubmenu2D("material", "data/textures/material.png");

        {
            ui::ButtonSubmenu2D* colors_submenu = new ui::ButtonSubmenu2D("colors", "data/textures/colors.png");

            {
                ui::ItemGroup2D* g_picker = new ui::ItemGroup2D("g_picker");
                g_picker->add_child(new ui::ColorPicker2D("color_picker", colors::RED));
                colors_submenu->add_child(g_picker);
            }

            {
                ui::ItemGroup2D* g_colors = new ui::ItemGroup2D("g_colors");
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_1", "data/textures/colors_template_1.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_2", "data/textures/colors_template_2.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_3", "data/textures/colors_template_3.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_4", "data/textures/colors_template_4.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_5", "data/textures/colors_template_5.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("color_template_palette_6", "data/textures/colors_template_6.png", ui::KEEP_RGB));
                g_colors->add_child(new ui::ButtonSubmenu2D("recent_colors", "data/textures/recent_colors.png", ui::KEEP_RGB));

                // TODO: missing specific color palettes..

                colors_submenu->add_child(g_colors);
            }

            material_submenu->add_child(colors_submenu);
        }

        {
            ui::ButtonSubmenu2D* mat_list_submenu = new ui::ButtonSubmenu2D("materials", "data/textures/material_samples.png");

            mat_list_submenu->add_child(new ui::TextureButton2D("save_material", "data/textures/submenu_mark.png"));
            mat_list_submenu->add_child(new ui::TextureButton2D("pick_material", "data/textures/pick_material.png", ui::ALLOW_TOGGLE));

            {
                ui::ItemGroup2D* g_material_samples = new ui::ItemGroup2D("g_material_samples");
                g_material_samples->add_child(new ui::TextureButton2D("aluminium", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                g_material_samples->add_child(new ui::TextureButton2D("charcoal", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                g_material_samples->add_child(new ui::TextureButton2D("rusted_iron", "data/textures/material_samples.png", ui::UNIQUE_SELECTION));
                mat_list_submenu->add_child(g_material_samples);
            }

            material_submenu->add_child(mat_list_submenu);
        }

        {
            ui::ButtonSubmenu2D* material_editor_submenu = new ui::ButtonSubmenu2D("material_editor", "data/textures/material_editor.png");

            {
                ui::ItemGroup2D* g_edit_pbr = new ui::ItemGroup2D("g_edit_pbr");
                g_edit_pbr->add_child(new ui::Slider2D("roughness", 0.7f));
                g_edit_pbr->add_child(new ui::Slider2D("metallic", 0.2f));
                material_editor_submenu->add_child(g_edit_pbr);
            }

            {
                ui::ItemGroup2D* g_edit_pattern = new ui::ItemGroup2D("g_edit_pattern");
                g_edit_pattern->add_child(new ui::Slider2D("noise_intensity", 0.0f, ui::SliderMode::VERTICAL, 0.0f, 10.0f));
                g_edit_pattern->add_child(new ui::Slider2D("noise_frequency", 20.0f, ui::SliderMode::VERTICAL, 0.0f, 50.0f));
                g_edit_pattern->add_child(new ui::Slider2D("noise_octaves", 8.0f, ui::SliderMode::VERTICAL, 0.0f, 16.0f, 1.0f));
                g_edit_pattern->add_child(new ui::ColorPicker2D("noise_color_picker", colors::WHITE));
                material_editor_submenu->add_child(g_edit_pattern);
            }

            material_submenu->add_child(material_editor_submenu);
        }

        root_2d->add_child(material_submenu);
    }

    {
        ui::ItemGroup2D* g_utilities = new ui::ItemGroup2D("g_utilities");

        {
            ui::ButtonSubmenu2D* mirror_submenu = new ui::ButtonSubmenu2D("mirror", "data/textures/mirror.png");
            mirror_submenu->add_child(new ui::TextureButton2D("mirror_toggle", "data/textures/mirror.png", ui::ALLOW_TOGGLE));
            ui::ItemGroup2D* g_mirror = new ui::ItemGroup2D("g_mirror");
            g_mirror->add_child(new ui::TextureButton2D("mirror_translation", "data/textures/mirror.png", ui::UNIQUE_SELECTION | ui::SELECTED));
            g_mirror->add_child(new ui::TextureButton2D("mirror_rotation", "data/textures/mirror.png", ui::UNIQUE_SELECTION));
            g_mirror->add_child(new ui::TextureButton2D("mirror_both", "data/textures/mirror.png", ui::UNIQUE_SELECTION));
            mirror_submenu->add_child(g_mirror);
            g_utilities->add_child(mirror_submenu);
        }

        g_utilities->add_child(new ui::TextureButton2D("snap_to_surface", "data/textures/snap_to_surface.png", ui::ALLOW_TOGGLE));
        g_utilities->add_child(new ui::TextureButton2D("snap_to_grid", "data/textures/snap_to_grid.png", ui::ALLOW_TOGGLE));

        {
            ui::ButtonSubmenu2D* lock_axis_submenu = new ui::ButtonSubmenu2D("lock_axis", "data/textures/lock_axis.png");
            lock_axis_submenu->add_child(new ui::TextureButton2D("lock_axis_toggle", "data/textures/lock_axis.png", ui::ALLOW_TOGGLE));
            ui::ItemGroup2D* g_lock_axis = new ui::ItemGroup2D("g_lock_axis");
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_x", "data/textures/x.png", ui::UNIQUE_SELECTION));
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_y", "data/textures/y.png", ui::UNIQUE_SELECTION));
            g_lock_axis->add_child(new ui::TextureButton2D("lock_axis_z", "data/textures/z.png", ui::UNIQUE_SELECTION | ui::SELECTED));
            lock_axis_submenu->add_child(g_lock_axis);
            g_utilities->add_child(lock_axis_submenu);
        }

        root_2d->add_child(g_utilities);
    }

    if (Renderer::instance->get_openxr_available()) {
        ui_3d = new Viewport3D(root_2d);
        entities.push_back(ui_3d);
    }

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    Node2D::clean();

    sculpt_editor.clean();
}

void RoomsEngine::update(float delta_time)
{
    Engine::update(delta_time);

    Node::check_controller_signals();

    if(ui_3d) {
        glm::mat4x4 pose = Input::get_controller_pose(HAND_LEFT, POSE_AIM);
        pose = glm::rotate(pose, glm::radians(-45.f), glm::vec3(1.0f, 0.0f, 0.0f));
        ui_3d->set_model(pose);
    }
    else {
        root_2d->update(delta_time);
    }

    for (auto entity : entities) {
        entity->update(delta_time);
    }

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

    if (!ui_3d) {
        root_2d->render();
    }
    
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
    sculpt_editor.set_sculpt_started(true);

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
