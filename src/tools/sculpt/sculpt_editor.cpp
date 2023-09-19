#include "sculpt_editor.h"
#include "sculpt.h"
#include "paint.h"
#include "graphics/raymarching_renderer.h"

#include "framework/scene/parse_scene.h"

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RaymarchingRenderer*>(Renderer::instance);

    sphere_mesh = parse_scene("data/meshes/wired_sphere.obj");
    sphere_mesh->set_material_color(colors::WHITE);
    cube_mesh = parse_scene("data/meshes/hollow_cube.obj");
    cube_mesh->set_material_color(colors::WHITE);

    mesh_preview = sphere_mesh;

    mirror_mesh = new EntityMesh();
    Mesh* quad_mesh = new Mesh();
    quad_mesh->create_quad(0.5f, 0.5f);
    mirror_mesh->set_material_diffuse(Texture::get("data/textures/mirror_quad_texture.png"));
    mirror_mesh->set_material_shader(Shader::get("data/shaders/mesh_texture.wgsl"));
    mirror_mesh->set_mesh(quad_mesh);

    floor_grid_mesh = new EntityMesh();
    Mesh* q_mesh = new Mesh();
    q_mesh->create_quad(3.0f, 3.0f);
    floor_grid_mesh->set_material_shader(Shader::get("data/shaders/mesh_grid.wgsl"));
    floor_grid_mesh->set_mesh(q_mesh);
    floor_grid_mesh->set_translation(glm::vec3(0.0f));
    floor_grid_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    mirror_gizmo.initialize(POSITION_GIZMO, sculpt_start_position + glm::vec3(0.0f, 1.0f, 0.0f));

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    // UI Layout from JSON
    {
        load_ui_layout( "data/ui.json" );
    }

    /*ui::Widget* debug = gui.make_rect({0, 0}, { 256.f, 144.f }, colors::RED);
    debug->priority = -1;*/

    // Set events
    {
        gui.bind("sculpt", [&](const std::string& signal, Color color) { enable_tool(SCULPT); });
        gui.bind("sphere", [&](const std::string& signal, Color color) {
            current_primitive = SD_SPHERE;
            mesh_preview = sphere_mesh;
        });
        gui.bind("cube", [&](const std::string& signal, Color color) {
            current_primitive = SD_BOX;
            mesh_preview = cube_mesh;
        });
        gui.bind("paint", [&](const std::string& signal, Color color) { enable_tool(PAINT); });
        gui.bind("mirror", [&](const std::string& signal, Color color) { use_mirror = !use_mirror; });
        gui.bind("snap_to_grid", [&](const std::string& signal, Color color) { snap_to_grid = !snap_to_grid; });

        // Bind all colors...

        auto on_press_color = [&](const std::string& signal, Color color) {
            current_color = color;
            add_recent_color(color);
        };

        auto& _colors = j_ui["colors_to_bind"];
        for (auto& color_name : _colors) {
            gui.bind(color_name, on_press_color);
        }

        // Bind recent colors...

        auto on_press_recent_color = [&](const std::string& signal, Color color) {
            current_color = color;
        };

        _colors = j_ui["recent_colors_to_bind"];
        for (auto& color_name : _colors) {
            gui.bind(color_name, on_press_recent_color);
        }

        max_recent_colors = _colors.size();
    }

    enable_tool(SCULPT);
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    bool tool_used = tools[current_tool]->update(delta_time);

    if (current_tool == SCULPT && tool_used) {
        sculpt_started = true;
    }

    Edit& edit_to_add = tools[current_tool]->get_edit_to_add();

    if (snap_to_grid) {
        float grid_multiplier = 1.0f / snap_grid_size;
        // Uncomment for grid size of half of the edit radius
        // grid_multiplier = 1.0f / (edit_to_add.dimensions.x / 2.0f);
        edit_to_add.position = glm::round(edit_to_add.position * grid_multiplier) / grid_multiplier;
    }

    // Set center of sculpture and reuse it as mirror center
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);

        mirror_origin = sculpt_start_position + glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Rotate the scene TODO: when ready move this out of tool to engine
    if (is_rotation_being_used()) {

        if (!rotation_started) {
            initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
            initial_hand_translation = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f);
        }

        rotation_diff = glm::inverse(initial_hand_rotation) * glm::inverse(Input::get_controller_rotation(HAND_LEFT));
        translation_diff = Input::get_controller_position(HAND_LEFT) - glm::vec3(0.0f, 1.0f, 0.0f) - initial_hand_translation;

        renderer->set_sculpt_rotation(sculpt_rotation * rotation_diff);
        renderer->set_sculpt_start_position(sculpt_start_position + translation_diff);

        rotation_started = true;
    }
    else {
        if (rotation_started) {
            sculpt_rotation = sculpt_rotation * rotation_diff;
            sculpt_start_position = sculpt_start_position + translation_diff;
            rotation_started = false;
            rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
        }
    }

    edit_to_add.primitive = current_primitive;
    edit_to_add.color = current_color;
    // ...

    // Set position of the preview edit
    renderer->set_preview_edit(edit_to_add);

    if (use_mirror) {
        mirror_origin = mirror_gizmo.update(mirror_origin);
    }

    if (tool_used) {
        renderer->push_edit(edit_to_add);

        // If the mirror is activated, mirror using the plane, and add another edit to the list
        if (use_mirror) {
            float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - mirror_origin);
            edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

            renderer->push_edit(edit_to_add);
        }
    }

    gui.update(delta_time);
}

void SculptEditor::render()
{
    Edit& edit_to_add = tools[current_tool]->get_edit_to_add();

#ifdef XR_SUPPORT
    // Render a hollowed edit
    mesh_preview->set_model(Input::get_controller_pose(gui.get_workspace().select_hand));
    mesh_preview->scale(edit_to_add.dimensions + glm::vec4(0.001f));
    mesh_preview->render();

    if (current_tool != NONE) {
        tools[current_tool]->render_scene();
        tools[current_tool]->render_ui();
    }

    gui.render();

    if (use_mirror) {
        mirror_gizmo.render();
        mirror_mesh->set_translation(mirror_origin);
        mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        mirror_mesh->render();
    }

    floor_grid_mesh->render();
#endif
}

void SculptEditor::enable_tool(eTool tool)
{
    if (current_tool != NONE) {
        tools[current_tool]->stop();
    }

    tools[tool]->start();
    current_tool = tool;
}

void SculptEditor::load_ui_layout(const std::string& filename)
{
    const json& j = load_json(filename);
    j_ui = j;
    float group_elements_pending = -1;

    float width = j["width"];
    float height = j["height"];
    gui.set_workspace({ width, height });

    std::function<void(const json&)> read_element = [&](const json& j) {

        std::string name = j["name"];
        std::string type = j["type"];

        if (type == "group")
        {
            assert(j.count("nitems") > 0);
            float nitems = j["nitems"];
            group_elements_pending = nitems;

            glm::vec4 color;
            if (j.count("color")) {
                color = load_vec4(j["color"]);
            }
            else {
                color = colors::GRAY;
            }

            ui::Widget* group = gui.make_group(name, nitems, color);

            if (j.count("store_widget"))
                widgets_loaded[name] = group;
        }
        else if (type == "button")
        {
            std::string texture = j["texture"];
            std::string shader = "data/shaders/mesh_texture_ui.wgsl";

            if (j.count("shader"))
                shader = j["shader"];

            Color color = colors::WHITE;
            if (j.count("color"))
                color = load_vec4(j["color"]);

            ui::Widget* widget = gui.make_button(name, texture.c_str(), shader.c_str(), color);
            group_elements_pending--;

            if (j.count("store_widget"))
                widgets_loaded[name] = widget;

            if (group_elements_pending == 0.f)
            {
                gui.close_group();
                group_elements_pending = -1;
            }
        }
        else if (type == "submenu")
        {
            ui::Widget* parent = widgets_loaded[name];
            gui.make_submenu(parent, name);

            assert(j.count("children") > 0);
            auto& _subelements = j["children"];
            for (auto& sub_el : _subelements) {
                read_element(sub_el);
            }

            gui.close_submenu();
        }
    };

    auto& _elements = j["elements"];
    for (auto& el : _elements) {
        read_element(el);
    }
}

void SculptEditor::add_recent_color(const Color& color)
{
    auto it = std::find(recent_colors.begin(), recent_colors.end(), color);

    // Color is not present in recents...
    if (it == recent_colors.end())
    {
        recent_colors.insert(recent_colors.begin(), color);

        if (recent_colors.size() > max_recent_colors)
        {
            recent_colors.pop_back();
        }
    }

    ui::Widget* recent_group = widgets_loaded["g_recent_colors"];

    assert(recent_colors.size() <= recent_group->children.size());
    for (uint8_t i = 0; i < recent_colors.size(); ++i)
    {
        ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(recent_group->children[i]);
        child->color = recent_colors[i];
    }
}
