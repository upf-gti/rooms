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

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    // Config UI
    const float size = 128.f;
    gui.set_workspace({ size, size });

    // UI Layout
    {
        gui.make_submenu("modes", { 24.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
        gui.make_button("smear", { 24.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/normal.png");
        gui.make_button("stamp", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
        gui.make_button("paint", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/stamp.png");
        /*gui.make_submenu("colorize", { 80.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sculpt_modes.png");
        gui.make_color_picker("colors", { edit_to_add.color.r, edit_to_add.color.g, edit_to_add.color.b, 1.0f }, { 48.f, 68.f }, { 32.f, 8.f });
        gui.close_submenu();*/
        gui.close_submenu();

        gui.make_submenu("primitives", { 52.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
        gui.make_button("sphere", { 38.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/sphere.png");
        gui.make_button("cube", { 66.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/cube.png");
        gui.close_submenu();

        gui.make_submenu("tools", { 80.f, 4.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/tools.png");
        gui.make_button("mirror", { 52.f, 36.f }, { 24.f, 24.f }, colors::WHITE, "data/textures/mirror.png");
        gui.close_submenu();
    }

    // Set events
    {
        // Modes
        gui.connect("colors", [&](const std::string& signal, const Color& color) {
            current_color = color;
        });

        // Primitives
        gui.connect("sphere", [&](const std::string& signal, float value) {
            current_primitive = SD_SPHERE;
            mesh_preview->set_mesh(Mesh::get("data/meshes/wired_sphere.obj"));
        });
        gui.connect("cube", [&](const std::string& signal, float value) {
            current_primitive = SD_BOX;
            mesh_preview->set_mesh(Mesh::get("data/meshes/hollow_cube.obj"));
        });

        // Tools
        gui.connect("mirror", [&](const std::string& signal, float value) {/* use_mirror = !use_mirror;*/ });
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

    // Set center of sculpture
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);
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

    // Set position of the preview edit
    renderer->set_preview_edit(edit_to_add);

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

    // Render a hollowed edit
    mesh_preview->set_model(Input::get_controller_pose(gui.get_workspace().select_hand));
    mesh_preview->scale(edit_to_add.dimensions + glm::vec4(0.001f));
    mesh_preview->render();

#ifdef XR_SUPPORT
    if (current_tool != NONE) {
        tools[current_tool]->render_scene();
        tools[current_tool]->render_ui();
    }
    gui.render();
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
