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
    mirror_mesh->set_mesh(quad_mesh);

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    // Config UI
    gui.set_workspace({ 256.f, 128.f });

    // UI Layout
    {
        gui.make_group("g_main_tools", 2, Color(0.4f));
        ui::Widget* primitives_widget = gui.make_button("primitives", "data/textures/cube.png", "data/textures/cube_selected.png");
        gui.make_button("paint", "data/textures/paint.png");
        gui.close_group();
        gui.make_button("mirror", "data/textures/mirror.png");
        ui::Widget* colors_widget = gui.make_button("colors", "data/textures/colors.png");

        gui.make_submenu(primitives_widget, "primitives");
            gui.make_group("g_primitives", 2, Color(0.4f));
            gui.make_button("sphere", "data/textures/sphere.png");
            gui.make_button("cube", "data/textures/cube.png");
            gui.close_group();
        gui.close_submenu();

        gui.make_submenu(colors_widget, "colors");
            gui.make_group("g_colors", 5, Color(0.4f));
            gui.make_button("color-template-1", "data/textures/colors_template.png");
            gui.make_button("color-template-2", "data/textures/colors_template.png");
            gui.make_button("color-template-3", "data/textures/colors_template.png");
            gui.make_button("color-template-4", "data/textures/colors_template.png");
            gui.make_button("color-template-5", "data/textures/colors_template.png");
            gui.close_group();
            gui.make_button("recent-colors", "data/textures/recent_colors.png");
        gui.close_submenu();

        // Old
        /*gui.make_submenu("modes", { 24.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
        gui.make_button("smear", { 24.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/normal.png");
        gui.make_button("stamp", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
        gui.make_button("paint", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
        gui.make_submenu("colorize", { 80.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
        gui.make_color_picker("colors", { current_color.r, current_color.g, current_color.b, 1.0f }, { 48.f, 68.f }, { 32.f, 8.f });
        gui.close_submenu();
        gui.close_submenu();

        gui.make_submenu("primitives", { 52.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
        gui.make_button("sphere", { 38.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
        gui.make_button("cube", { 66.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/cube.png");
        gui.close_submenu();

        gui.make_submenu("tools", { 80.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/tools.png");
        gui.make_button("mirror", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/mirror.png");
        gui.close_submenu();*/
    }

    // Set events
    {
        // Primitives
        gui.bind("sphere", [&](const std::string& signal, float value) {
            current_primitive = SD_SPHERE;
            mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
        });
        gui.bind("cube", [&](const std::string& signal, float value) {
            current_primitive = SD_BOX;
            mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj"));
        });

        // Tools

        /*gui.bind("colors", [&](const std::string& signal, const Color& color) {
            current_color = color;
        });*/

        gui.bind("mirror", [&](const std::string& signal, float value) { use_mirror = !use_mirror; });
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

    // Set center of sculpture and reuse it as mirror center
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);

        mirror_gizmo.initialize(POSITION_GIZMO);

        mirror_origin = sculpt_start_position;
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

    if (tool_used) {
        renderer->push_edit(edit_to_add);

        // If the mirror is activated, mirror using the plane, and add another edit to the list
        if (use_mirror) {
            float dist_to_plane = glm::dot(mirror_normal, edit_to_add.position - sculpt_start_position + mirror_origin);
            edit_to_add.position = edit_to_add.position - mirror_normal * dist_to_plane * 2.0f;

            renderer->push_edit(edit_to_add);
        }
    }

    gui.update(delta_time);

    if (use_mirror) {
        mirror_origin = mirror_gizmo.update(mirror_origin);
    }
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
