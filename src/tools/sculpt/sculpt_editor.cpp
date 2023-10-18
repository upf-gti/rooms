#include "sculpt_editor.h"
#include "sculpt.h"
#include "paint.h"
#include "sweep.h"
#include "graphics/renderers/rooms_renderer.h"
#include "framework/scene/parse_scene.h"

#include "graphics/renderer_storage.h"

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    sphere_mesh = parse_scene("data/meshes/wired_sphere.obj");
    sphere_mesh->set_material_color(colors::WHITE);
    cube_mesh = parse_scene("data/meshes/hollow_cube.obj");
    cube_mesh->set_material_color(colors::WHITE);

    mesh_preview = sphere_mesh;

    mirror_mesh = new EntityMesh();
    mirror_mesh->set_material_diffuse(RendererStorage::get_texture("data/textures/mirror_quad_texture.png"));
    mirror_mesh->set_material_shader(RendererStorage::get_shader("data/shaders/mesh_texture.wgsl"));
    mirror_mesh->set_material_flag(MATERIAL_TRANSPARENT);
    mirror_mesh->set_mesh(RendererStorage::get_mesh("quad"));
    mirror_mesh->scale(glm::vec3(0.5f));

    floor_grid_mesh = new EntityMesh();
    floor_grid_mesh->set_material_shader(RendererStorage::get_shader("data/shaders/mesh_grid.wgsl"));
    floor_grid_mesh->set_mesh(RendererStorage::get_mesh("quad"));
    floor_grid_mesh->set_translation(glm::vec3(0.0f));
    floor_grid_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    floor_grid_mesh->scale(glm::vec3(3.f));

    mirror_gizmo.initialize(POSITION_GIZMO, sculpt_start_position);

    tools[SCULPT] = new SculptTool();
    tools[PAINT] = new PaintTool();
    tools[SWEEP] = new SweepTool();

    dynamic_cast<SweepTool*>(tools[SWEEP])->set_sculpt_editor(this);

    for (auto& tool : tools) {
        if (tool) {
            tool->initialize();
        }
    }

    // UI Layout from JSON
    {
        gui.load_layout( "data/ui/main.json" );
    }

    // Set events
    {
        gui.bind("sculpt", [&](const std::string& signal, void* button) { enable_tool(SCULPT); });
        gui.bind("paint", [&](const std::string& signal, void* button) { enable_tool(PAINT); });

        gui.bind("sphere", [&](const std::string& signal, void* button) {  set_primitive(SD_SPHERE, sphere_mesh); });
        gui.bind("cube", [&](const std::string& signal, void* button) { set_primitive(SD_BOX, cube_mesh); });
        gui.bind("cone", [&](const std::string& signal, void* button) { set_primitive(SD_CONE); });
        gui.bind("capsule", [&](const std::string& signal, void* button) { set_primitive(SD_CAPSULE); });
        gui.bind("cylinder", [&](const std::string& signal, void* button) { set_primitive(SD_CYLINDER); });
        gui.bind("torus", [&](const std::string& signal, void* button) { set_primitive(SD_TORUS); });

        gui.bind("onion", [&](const std::string& signal, void* button) { set_primitive_modifier(onion_enabled); });
        gui.bind("capped", [&](const std::string& signal, void* button) { set_primitive_modifier(capped_enabled); });

        gui.bind("mirror", [&](const std::string& signal, void* button) { use_mirror = !use_mirror; });
        gui.bind("snap_to_grid", [&](const std::string& signal, void* button) { enable_tool(SWEEP);});//snap_to_grid = !snap_to_grid; });

        // Bind recent color buttons...

        ui::UIEntity* recent_group = gui.get_widget_from_name("g_recent_colors");
        if (!recent_group){
            assert(0);
            std::cerr << "Cannot find recent_colors button group!" << std::endl;
            return;
        }

        // Bind colors callback...

        for (auto it : gui.get_widgets())
        {
            if (it.second->type != ui::BUTTON) continue;
            ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(it.second);
            if (child->is_color_button) {
                gui.bind(child->signal, [&](const std::string& signal, void* button) {
                    const Color& color = (static_cast<ui::ButtonWidget*>(button))->color;
                    current_color = color;
                    add_recent_color(color);
                });
            }
        }

        max_recent_colors = recent_group->get_children().size();
        for (size_t i = 0; i < max_recent_colors; ++i)
        {
            ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(recent_group->get_children()[i]);
            gui.bind(child->signal, [&](const std::string& signal, void* button) {
                current_color = (static_cast<ui::ButtonWidget*>(button))->color;
            });
        }
    }

    // Create helper ui
    {
        helper_gui.load_layout("data/ui/helper.json");

        // Customize a little bit...
        helper_gui.get_workspace().hand = HAND_RIGHT;
        helper_gui.get_workspace().root_pose = POSE_GRIP;
    }

    enable_tool(SCULPT);
}

void SculptEditor::clean()
{
    if (mirror_mesh) {
        delete mirror_mesh;
    }

    if (floor_grid_mesh) {
        delete floor_grid_mesh;
    }
}

void SculptEditor::update(float delta_time)
{
    if (current_tool == NONE) {
        return;
    }

    preview_tmp_edits.clear();
    new_edits.clear();

    Tool& tool_used = *tools[current_tool];

    if (Input::was_button_pressed(XR_BUTTON_B))
        stamp_enabled = !stamp_enabled;

    // Update tool properties...
    tool_used.stamp = stamp_enabled;
    // ...

    bool is_tool_used = tool_used.update(delta_time);

    if (current_tool == SCULPT && is_tool_used) {
        sculpt_started = true;
    }

    Edit& edit_to_add = tool_used.get_edit_to_add();

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
        mirror_origin = sculpt_start_position;
    }

    // Rotate the scene
    if (is_rotation_being_used()) {

        if (!rotation_started) {
            initial_hand_rotation = glm::inverse(Input::get_controller_rotation(HAND_LEFT));
            initial_hand_translation = Input::get_controller_position(HAND_LEFT);
        }

        rotation_diff = glm::inverse(initial_hand_rotation) * glm::inverse(Input::get_controller_rotation(HAND_LEFT));
        translation_diff = Input::get_controller_position(HAND_LEFT) - initial_hand_translation;

        renderer->set_sculpt_rotation(sculpt_rotation * rotation_diff);
        renderer->set_sculpt_start_position(sculpt_start_position + translation_diff);

        rotation_started = true;
    }
    else if (rotation_started) {
        sculpt_rotation = sculpt_rotation * rotation_diff;
        sculpt_start_position = sculpt_start_position + translation_diff;
        rotation_started = false;
        rotation_diff = { 0.0f, 0.0f, 0.0f, 1.0f };
    }

    // Update edit dimensions

    if (capped_enabled)
    {
        float multiplier = -Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 2.f;
        capped_value = glm::clamp(multiplier + capped_value, -1.f, 1.f);
    }
    else if (onion_enabled)
    {
        float multiplier = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 1.f;
        onion_thickness = glm::clamp(multiplier + onion_thickness, 0.f, 1.f);
    }
    else
    {
        float size_multiplier = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 0.1f;
        glm::vec3 new_dimensions = glm::clamp(size_multiplier + glm::vec3(edit_to_add.dimensions), 0.001f, 0.1f);
        edit_to_add.dimensions = glm::vec4(new_dimensions, edit_to_add.dimensions.w);

        // Update primitive specific size
        size_multiplier = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1f;
        edit_to_add.dimensions.w = glm::clamp(size_multiplier + edit_to_add.dimensions.w, 0.001f, 0.1f);
    }

    // Update current edit properties...
    edit_to_add.primitive = current_primitive;
    edit_to_add.color = current_color;
    edit_to_add.parameters.x = onion_thickness;
    edit_to_add.parameters.y = capped_value;

    if (is_tool_used) {
        new_edits.push_back(edit_to_add);
    }

    if (renderer->get_openxr_available()) {
        preview_tmp_edits.push_back(edit_to_add);
    }

    // Mirror functionality
    if (use_mirror) {
        mirror_origin = mirror_gizmo.update(mirror_origin, delta_time);

        uint32_t preview_edit_count = preview_tmp_edits.size();
        for (uint32_t i = 0u; i < preview_edit_count; i++) {
            Edit inverted_preview_edit = preview_tmp_edits[i];
            float dist_to_plane = glm::dot(mirror_normal, inverted_preview_edit.position - mirror_origin);
            inverted_preview_edit.position = inverted_preview_edit.position - mirror_normal * dist_to_plane * 2.0f;

            preview_tmp_edits.push_back(inverted_preview_edit);
        }

        uint32_t edit_count = new_edits.size();
        for (uint32_t i = 0u; i < edit_count; i++) {
            Edit inverted_edit = new_edits[i];
            float dist_to_plane = glm::dot(mirror_normal, inverted_edit.position - mirror_origin);
            inverted_edit.position = inverted_edit.position - mirror_normal * dist_to_plane * 2.0f;

            new_edits.push_back(inverted_edit);
        }
    }

    gui.update(delta_time);
    helper_gui.update(delta_time);

    // Push to the renderer the edits and the previews
    renderer->push_preview_edit_list(preview_tmp_edits);
    renderer->push_edit_list(new_edits);
}

void SculptEditor::render()
{
    Tool& tool_used = *tools[current_tool];
    Edit& edit_to_add = tool_used.get_edit_to_add();

#ifdef XR_SUPPORT

    if (mesh_preview)
    {
        // Render a hollowed edit
        mesh_preview->set_model(Input::get_controller_pose(gui.get_workspace().select_hand));
        mesh_preview->scale(edit_to_add.dimensions + glm::vec4(0.001f));
        mesh_preview->render();
    }

    if (current_tool != NONE) {
        tool_used.render_scene();
        tool_used.render_ui();
    }

    gui.render();
    helper_gui.render();

    if (use_mirror) {
        mirror_gizmo.render();
        mirror_mesh->set_translation(mirror_origin);
        mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        mirror_mesh->render();
    }

    floor_grid_mesh->render();
#endif
}

void SculptEditor::set_sculpt_started(bool value)
{
    sculpt_started = true;
}

void SculptEditor::set_primitive(sdPrimitive primitive, EntityMesh* preview)
{
    current_primitive = primitive;
    mesh_preview = preview;
}

void SculptEditor::set_primitive_modifier(bool& modifier)
{
    const bool last_value = modifier;

    // Disable all
    capped_enabled  = false;
    onion_enabled   = false;

    // Enable specific item
    modifier = !last_value;
}

void SculptEditor::enable_tool(eTool tool)
{
    if (current_tool != NONE) {
        tools[current_tool]->stop();
    }

    tools[tool]->start();
    current_tool = tool;

    switch (tool)
    {
    case SCULPT:
        helper_gui.change_list_layout("sculpt");
        break;
    case PAINT:
        helper_gui.change_list_layout("paint");
        break;
    default:
        break;
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

    ui::UIEntity* recent_group = gui.get_widget_from_name("g_recent_colors");

    assert(recent_colors.size() <= recent_group->get_children().size());
    for (uint8_t i = 0; i < recent_colors.size(); ++i)
    {
        ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(recent_group->get_children()[i]);
        child->color = recent_colors[i];
    }
}
