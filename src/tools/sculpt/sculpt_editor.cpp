#include "sculpt_editor.h"
#include "sculpt.h"
#include "paint.h"
#include "graphics/raymarching_renderer.h"

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RaymarchingRenderer*>(Renderer::instance);

    mesh_preview = new EntityMesh();
    mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
    mesh_preview->set_shader(Shader::get("data/shaders/mesh_color.wgsl"));
    mesh_preview->set_color(colors::WHITE);

    mirror_mesh = new EntityMesh();
    Mesh* quad_mesh = new Mesh();
    quad_mesh->create_quad(0.5f, 0.5f);
    quad_mesh->set_texture(Texture::get("data/textures/mirror_quad_texture.png"));
    mirror_mesh->set_shader(Shader::get("data/shaders/mesh_texture.wgsl"));
    mirror_mesh->set_mesh(quad_mesh);

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    // Config UI
    gui.set_workspace({ 256.f, 144 });

    // UI Layout
    {
        // debug
        /*ui::Widget* debugw = gui.make_rect({0, 0}, { 256.f, 144.f }, colors::RED);
        debugw->priority = -1;*/
        // ...

        gui.make_group("g_main_tools", 2, colors::GRAY);
        ui::Widget* sculpt_widget = gui.make_button("sculpt", "data/textures/cube.png");
        ui::Widget* paint_widget = gui.make_button("paint", "data/textures/paint.png");
        gui.close_group();
        gui.make_button("mirror", "data/textures/mirror.png");
        gui.make_button("stamp", "data/textures/stamp.png");
        ui::Widget* colors_widget = gui.make_button("colors", "data/textures/colors.png");

        // Sculpt primitive dropdown
        gui.make_submenu(sculpt_widget, "sculpt");
            gui.make_group("g0_primitives", 2, colors::GRAY);
            gui.make_button("sphere", "data/textures/sphere.png");
            gui.make_button("cube", "data/textures/cube.png");
            gui.close_group();
        gui.close_submenu();

        // Paint primitive dropdown
        gui.make_submenu(paint_widget, "paint");
            gui.make_group("g1_primitives", 2, colors::GRAY);
            gui.make_button("sphere", "data/textures/sphere.png");
            gui.make_button("cube", "data/textures/cube.png");
            gui.close_group();
        gui.close_submenu();

        /*
        *   Colors ...
        */

        gui.make_submenu(colors_widget, "colors");
            gui.make_group("g_colors", 5, colors::GRAY);
            ui::Widget* color_template_palette_1 = gui.make_button("color_template_palette_1", "data/textures/colors_template_1.png");
            ui::Widget* color_template_palette_2 = gui.make_button("color_template_palette_2", "data/textures/colors_template_2.png");
            ui::Widget* color_template_palette_3 = gui.make_button("color_template_palette_3", "data/textures/colors_template_3.png");
            ui::Widget* color_template_palette_4 = gui.make_button("color_template_palette_4", "data/textures/colors_template_4.png");
            ui::Widget* color_template_palette_5 = gui.make_button("color_template_palette_5", "data/textures/colors_template_5.png");
            gui.close_group();
            gui.make_button("recent-colors", "data/textures/recent_colors.png");
        gui.close_submenu();

        gui.make_submenu(color_template_palette_1, "color_template_palette_1");
        gui.make_group("g_colors_t1", 4, colors::GRAY);
        gui.make_button("colors_t1_1", "data/textures/circle256.png", "data/shaders/mesh_texture.wgsl", Color(0.2f, 0.21f, 0.77f, 1.f));
        gui.make_button("colors_t1_2", "data/textures/circle256.png", "data/shaders/mesh_texture.wgsl", Color(0.41f, 0.57f, 0.79f, 1.f));
        gui.make_button("colors_t1_3", "data/textures/circle256.png", "data/shaders/mesh_texture.wgsl", Color(0.41f, 0.76f, 0.79f, 1.f));
        gui.make_button("colors_t1_4", "data/textures/circle256.png", "data/shaders/mesh_texture.wgsl", Color(0.64f, 0.9f, 0.93f,  1.f));
        gui.close_group();
        gui.close_submenu();
    }

    // Set events
    {
        gui.bind("sculpt", [&](const std::string& signal, float value) { enable_tool(SCULPT); });
        gui.bind("sphere", [&](const std::string& signal, float value) {
            current_primitive = SD_SPHERE;
            mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
        });
        gui.bind("cube", [&](const std::string& signal, float value) {
            current_primitive = SD_BOX;
            mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj"));
        });
        gui.bind("paint", [&](const std::string& signal, float value) { enable_tool(PAINT); });
        gui.bind("mirror", [&](const std::string& signal, float value) { use_mirror = !use_mirror; });
        gui.bind("stamp", [&](const std::string& signal, float value) { stamp_enabled = !stamp_enabled; });

        gui.bind("colors_t1_1", [&](const std::string& signal, float value) { current_color = Color(0.2f,  0.21f, 0.77f,  1.f); });
        gui.bind("colors_t1_2", [&](const std::string& signal, float value) { current_color = Color(0.41f, 0.57f, 0.79f,  1.f); });
        gui.bind("colors_t1_3", [&](const std::string& signal, float value) { current_color = Color(0.41f, 0.76f, 0.79f,  1.f); });
        gui.bind("colors_t1_4", [&](const std::string& signal, float value) { current_color = Color(0.64f, 0.9f,  0.93f,  1.f); });
    }

    enable_tool(SCULPT);
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    tools[current_tool]->stamp_enabled = stamp_enabled;

    bool tool_used = tools[current_tool]->update(delta_time);

    if (current_tool == SCULPT && tool_used) {
        sculpt_started = true;
    }

    Edit& edit_to_add = tools[current_tool]->get_edit_to_add();

    // Set center of sculpture and reuse it as mirror center
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);

        mirror_gizmo.initialize(POSITION_GIZMO, sculpt_start_position + glm::vec3(0.0f, 1.0f, 0.0f));

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
