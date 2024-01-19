#include "sculpt_editor.h"

#include "sculpt.h"
#include "paint.h"
#include "sweep.h"

#include "includes.h"

#include "framework/entities/entity_ui.h"
#include "framework/input.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "backends/imgui_impl_wgpu.h"
#include "backends/imgui_impl_glfw.h"

#include "spdlog/spdlog.h"

void SculptEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    mirror_mesh = new EntityMesh();
    mirror_mesh->add_surface(RendererStorage::get_surface("quad"));
    mirror_mesh->scale(glm::vec3(0.5f));

    Material mirror_material;
    mirror_material.shader = RendererStorage::get_shader("data/shaders/mesh_texture.wgsl");
    mirror_material.diffuse_texture = RendererStorage::get_texture("data/textures/mirror_quad_texture.png");
    mirror_material.flags |= MATERIAL_DIFFUSE | MATERIAL_TRANSPARENT;

    mirror_mesh->set_surface_material_override(mirror_mesh->get_surface(0), mirror_material);

    floor_grid_mesh = new EntityMesh();
    floor_grid_mesh->add_surface(RendererStorage::get_surface("quad"));
    floor_grid_mesh->set_translation(glm::vec3(0.0f));
    floor_grid_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    floor_grid_mesh->scale(glm::vec3(3.f));

    Material grid_material;
    grid_material.shader = RendererStorage::get_shader("data/shaders/mesh_grid.wgsl");
    grid_material.flags |= MATERIAL_TRANSPARENT;

    floor_grid_mesh->set_surface_material_override(mirror_mesh->get_surface(0), grid_material);

    axis_lock_gizmo.initialize(POSITION_GIZMO, sculpt_start_position);
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

    // Edit preview mesh
    {
        Surface* sphere_surface = new Surface();
        sphere_surface->create_sphere();

        mesh_preview = new EntityMesh();
        mesh_preview->add_surface(sphere_surface);

        Material preview_material;
        preview_material.shader = RendererStorage::get_shader("data/shaders/mesh_transparent.wgsl");
        preview_material.flags |= MATERIAL_TRANSPARENT;
        preview_material.priority = 1;

        mesh_preview->set_surface_material_override(sphere_surface, preview_material);

        mesh_preview_outline = new EntityMesh();
        mesh_preview_outline->add_surface(sphere_surface);

        Material outline_material;
        outline_material.shader = RendererStorage::get_shader("data/shaders/mesh_outline.wgsl");

        mesh_preview_outline->set_surface_material_override(sphere_surface, outline_material);
    }

    // Add pbr materials data 
    {
        add_pbr_material_data("aluminium",  Color(0.912f, 0.914f, 0.92f, 1.0f),     0.0f,   1.0f);
        add_pbr_material_data("charcoal",   Color(0.02f, 0.02f, 0.02f, 1.0f),       0.5f,   0.0f);
        add_pbr_material_data("copper",     Color(0.926f, 0.721f, 0.504f, 1.0f),    0.0f,   1.0f);
        add_pbr_material_data("concrete",   Color(0.51f, 0.51f, 0.51f, 1.0f),       0.5f,   0.0f);
        add_pbr_material_data("rusted_iron",Color(0.531f, 0.512f, 0.496f, 1.0f),    0.0f,   1.0f);
    }

    // Load ui and Bind callbacks
    bind_events();

    enable_tool(SCULPT);

    renderer->change_stroke(stroke_parameters);
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

    // Update ui/vr controller actions
    gui.update(delta_time);
    helper_gui.update(delta_time);

    Tool& tool_used = *tools[current_tool];

    // Update tool properties...
    tool_used.stamp = stamp_enabled;
    // ...

    bool is_tool_used = tool_used.update(delta_time, stroke_parameters);

    Edit& edit_to_add = tool_used.get_edit_to_add();

    if (Input::was_key_pressed(GLFW_KEY_U) || Input::was_grab_pressed(HAND_LEFT)) {
        renderer->undo();
    }

    if (Input::was_key_pressed(GLFW_KEY_R) || Input::was_grab_pressed(HAND_RIGHT)) {
        renderer->redo();
    }

    if (snap_to_grid) {
        float grid_multiplier = 1.f / snap_grid_size;
        // Uncomment for grid size of half of the edit radius
        // grid_multiplier = 1.f / (edit_to_add.dimensions.x / 2.f);
        edit_to_add.position = glm::round(edit_to_add.position * grid_multiplier) / grid_multiplier;
    }

    // Set center of sculpture and reuse it as mirror center
    if (!sculpt_started) {
        sculpt_start_position = edit_to_add.position;
        renderer->set_sculpt_start_position(sculpt_start_position);
        mirror_origin = sculpt_start_position;
        axis_lock_position = sculpt_start_position;
    }

    if (axis_lock) {

        axis_lock_position = axis_lock_gizmo.update(axis_lock_position, delta_time);

        glm::vec3 locked_pos = edit_to_add.position;

        if (axis_lock_mode & AXIS_LOCK_X)
            locked_pos.x = axis_lock_position.x;
        else if (axis_lock_mode & AXIS_LOCK_Y)
            locked_pos.y = axis_lock_position.y;
        else if (axis_lock_mode & AXIS_LOCK_Z)
            locked_pos.z = axis_lock_position.z;

        edit_to_add.position = locked_pos;
        edit_to_add.rotation = glm::quat();
    }

    if (current_tool == SCULPT && is_tool_used) {
        sculpt_started = true;
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
        translation_diff = {};
    }

    // Update edit dimensions
    {
        float size_multiplier = Input::get_thumbstick_value(HAND_RIGHT).y * delta_time * 0.1f;
        dimensions_dirty |= (fabsf(size_multiplier) > 0.f);
        glm::vec3 new_dimensions = glm::clamp(size_multiplier + glm::vec3(edit_to_add.dimensions), 0.001f, 0.1f);
        edit_to_add.dimensions = glm::vec4(new_dimensions, edit_to_add.dimensions.w);

        // Update primitive specific size
        size_multiplier = Input::get_thumbstick_value(HAND_LEFT).y * delta_time * 0.1f;
        edit_to_add.dimensions.w = glm::clamp(size_multiplier + edit_to_add.dimensions.w, 0.001f, 0.1f);
        dimensions_dirty |= (fabsf(size_multiplier) > 0.f);
    }

    // Update current edit properties...

    // Push edits in 3d texture space
    edit_to_add.position -= (sculpt_start_position + translation_diff);
    edit_to_add.position = (sculpt_rotation * rotation_diff) * edit_to_add.position;
    edit_to_add.rotation *= (sculpt_rotation * rotation_diff);

    // if any parameter changed or just stopped sculpting
    if (stroke_parameters.is_dirty() || (was_tool_used && !is_tool_used)) {
        renderer->change_stroke(stroke_parameters);
        stroke_parameters.set_dirty(false);
    }

    if (is_tool_used) {
        new_edits.push_back(edit_to_add);
        // Add recent color only when is used...
        add_recent_color(stroke_parameters.get_material().color);
    }

    if (renderer->get_openxr_available()) {
        preview_tmp_edits.push_back(edit_to_add);
    }

    // Mirror functionality
    if (use_mirror) {
        mirror_origin = mirror_gizmo.update(mirror_origin, delta_time);

        uint64_t preview_edit_count = preview_tmp_edits.size();
        for (uint64_t i = 0u; i < preview_edit_count; i++) {
            Edit inverted_preview_edit = preview_tmp_edits[i];
            float dist_to_plane = glm::dot(mirror_normal, inverted_preview_edit.position - mirror_origin);
            inverted_preview_edit.position = inverted_preview_edit.position - mirror_normal * dist_to_plane * 2.0f;

            preview_tmp_edits.push_back(inverted_preview_edit);
        }

        uint64_t edit_count = new_edits.size();
        for (uint64_t i = 0u; i < edit_count; i++) {
            Edit inverted_edit = new_edits[i];
            float dist_to_plane = glm::dot(mirror_normal, inverted_edit.position - mirror_origin);
            inverted_edit.position = inverted_edit.position - mirror_normal * dist_to_plane * 2.0f;

            new_edits.push_back(inverted_edit);
        }
    }

    // TODO tmp preview stroke
    renderer->set_preview_edit(edit_to_add);

    // Push to the renderer the edits and the previews
    renderer->push_preview_edit_list(preview_tmp_edits);
    renderer->push_edit_list(new_edits);

    was_tool_used = is_tool_used;
}

void SculptEditor::render()
{
    Tool& tool_used = *tools[current_tool];
    Edit& edit_to_add = tool_used.get_edit_to_add();

    if (mesh_preview && renderer->get_openxr_available())
    {
        update_edit_preview(edit_to_add.dimensions);

        // Render something to be able to cull faces later...
        if (stroke_parameters.get_operation() == OP_SUBSTRACTION || stroke_parameters.get_operation() == OP_SMOOTH_SUBSTRACTION || 
            stroke_parameters.get_operation() == OP_PAINT || stroke_parameters.get_operation() == OP_SMOOTH_PAINT)
        {
                mesh_preview->render();
        }
        else
        {
            mesh_preview_outline->set_model(mesh_preview->get_model());
            mesh_preview_outline->render();
        }
    }

    if (current_tool != NONE) {
        tool_used.render_scene();
        tool_used.render_ui();
    }

    if (renderer->get_openxr_available()) {
        gui.render();
        helper_gui.render();
    }

    if (axis_lock) {
        axis_lock_gizmo.render();

        mirror_mesh->set_translation(axis_lock_position);
        if (axis_lock_mode & AXIS_LOCK_X)
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        else if (axis_lock_mode & AXIS_LOCK_Y)
            mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        // debug
        if (Input::was_key_pressed(GLFW_KEY_X))
            axis_lock_mode = AXIS_LOCK_X;
        if (Input::was_key_pressed(GLFW_KEY_Y))
            axis_lock_mode = AXIS_LOCK_Y;
        if (Input::was_key_pressed(GLFW_KEY_Z))
            axis_lock_mode = AXIS_LOCK_Z;

        mirror_mesh->render();
    }
    else if (use_mirror) {
        mirror_gizmo.render();
        mirror_mesh->set_translation(mirror_origin);
        mirror_mesh->rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        mirror_mesh->render();
    }

    floor_grid_mesh->render();
}

void SculptEditor::render_gui()
{
    StrokeMaterial& stroke_material = stroke_parameters.get_material();

    bool changed = false;

    ImGui::Text("Material");

    ImGui::Separator();

    ImGui::Text("PBR");
    changed |= ImGui::ColorEdit4("Base Color", &stroke_material.color[0]);
    changed |= ImGui::SliderFloat("Roughness", &stroke_material.roughness, 0.f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &stroke_material.metallic, 0.f, 1.0f);

    ImGui::Separator();

    ImGui::Text("Noise");
    changed |= ImGui::ColorEdit4("Color", &stroke_material.noise_color[0]);
    changed |= ImGui::SliderFloat("Intensity", &stroke_material.noise_params.x, 0.f, 1.0f);
    changed |= ImGui::SliderFloat("Frequency", &stroke_material.noise_params.y, 0.f, 50.0f);

    int tmp = static_cast<int>(stroke_material.noise_params.z);
    changed |= ImGui::SliderInt("Octaves", &tmp, 1, 16);
    stroke_material.noise_params.z = static_cast<float>(tmp);

    if (changed)
        stroke_parameters.set_dirty(true);
}

void SculptEditor::update_edit_preview(const glm::vec4& dims)
{
    // Recreate mesh depending on primitive parameters

    if (dimensions_dirty)
    {
        // Expand a little bit the edges
        glm::vec4 grow_dims = dims * 1.01f;

        switch (stroke_parameters.get_primitive())
        {
        case SD_SPHERE:
            mesh_preview->get_surface(0)->create_sphere(grow_dims.x);
            break;
        case SD_BOX:
            mesh_preview->get_surface(0)->create_rounded_box(grow_dims.x, grow_dims.y, grow_dims.z, (dims.w / 0.1f) * grow_dims.x);
            break;
        case SD_CONE:
            mesh_preview->get_surface(0)->create_cone(grow_dims.w, grow_dims.x);
            mesh_preview->rotate(glm::radians(-90.f), { 1.f, 0.f, 0.f });
            break;
        case SD_CYLINDER:
            mesh_preview->get_surface(0)->create_cylinder(grow_dims.w, grow_dims.x);
            mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
            mesh_preview->translate({ 0.f, -dims.x * 0.5f, 0.f });
            break;
        case SD_CAPSULE:
            mesh_preview->get_surface(0)->create_capsule(grow_dims.w, grow_dims.x);
            mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
            mesh_preview->translate({ 0.f, -dims.x * 0.5f, 0.f });
            break;
        case SD_TORUS:
            mesh_preview->get_surface(0)->create_torus(grow_dims.x, glm::clamp(grow_dims.w, 0.0001f, grow_dims.x));
            mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
            break;
        default:
            break;
        }

        spdlog::trace("Edit mesh preview generated!");

        dimensions_dirty = false;
    }

    mesh_preview->set_model(Input::get_controller_pose(gui.get_workspace().select_hand));

    // Update model depending on the primitive
    switch (stroke_parameters.get_primitive())
    {
    case SD_CONE:
        mesh_preview->rotate(glm::radians(-90.f), { 1.f, 0.f, 0.f });
        break;
    case SD_CYLINDER:
        mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
        mesh_preview->translate({ 0.f, -dims.x * 0.5f, 0.f });
        break;
    case SD_CAPSULE:
        mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
        mesh_preview->translate({ 0.f, -dims.x * 0.5f, 0.f });
        break;
    case SD_TORUS:
        mesh_preview->rotate(glm::radians(90.f), { 1.f, 0.f, 0.f });
        break;
    default:
        break;
    }
}

void SculptEditor::set_sculpt_started(bool value)
{
    sculpt_started = true;
}

void SculptEditor::set_primitive(sdPrimitive primitive)
{
    stroke_parameters.set_primitive(primitive);
    dimensions_dirty = true;
}

void SculptEditor::toggle_onion_modifier()
{
    capped_enabled = false;
    onion_enabled = !onion_enabled;

    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.x = onion_enabled ? onion_thickness : 0.f;

    stroke_parameters.set_parameters(parameters);
}

void SculptEditor::toggle_capped_modifier()
{
    onion_enabled = false;
    capped_enabled = !capped_enabled;

    glm::vec4 parameters = stroke_parameters.get_parameters();
    parameters.y = capped_enabled ? capped_value : -1.f;

    stroke_parameters.set_parameters(parameters);
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

bool SculptEditor::is_rotation_being_used() {
    return Input::get_trigger_value(HAND_LEFT) > 0.5;
}


void SculptEditor::bind_events()
{
    // UI Layout from JSON
    {
        gui.load_layout("data/ui/main.json");
    }

    // Set events
    {
        gui.bind("sculpt", [&](const std::string& signal, void* button) { enable_tool(SCULPT); });
        gui.bind("paint", [&](const std::string& signal, void* button) { enable_tool(PAINT); });

        gui.bind("sphere", [&](const std::string& signal, void* button) {  set_primitive(SD_SPHERE); });
        gui.bind("cube", [&](const std::string& signal, void* button) { set_primitive(SD_BOX); });
        gui.bind("cone", [&](const std::string& signal, void* button) { set_primitive(SD_CONE); });
        gui.bind("capsule", [&](const std::string& signal, void* button) { set_primitive(SD_CAPSULE); });
        gui.bind("cylinder", [&](const std::string& signal, void* button) { set_primitive(SD_CYLINDER); });
        gui.bind("torus", [&](const std::string& signal, void* button) { set_primitive(SD_TORUS); });

        gui.bind("onion", [&](const std::string& signal, void* button) { toggle_onion_modifier(); });
        gui.bind("onion_value", [&](const std::string& signal, float value) { onion_thickness = glm::clamp(value, 0.f, 1.f); });
        gui.bind("capped", [&](const std::string& signal, void* button) { toggle_capped_modifier(); });
        gui.bind("cap_value", [&](const std::string& signal, float value) { capped_value = glm::clamp(value * 2.f - 1.f, -1.f, 1.f); });

        gui.bind("mirror", [&](const std::string& signal, void* button) { use_mirror = !use_mirror; });
        gui.bind("snap_to_grid", [&](const std::string& signal, void* button) { snap_to_grid = !snap_to_grid; });
        gui.bind("lock_axis_toggle", [&](const std::string& signal, void* button) { axis_lock = !axis_lock; });
        gui.bind("lock_axis_x", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_X; });
        gui.bind("lock_axis_y", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Y; });
        gui.bind("lock_axis_z", [&](const std::string& signal, void* button) { axis_lock_mode = AXIS_LOCK_Z; });

        gui.bind("pbr_roughness", [&](const std::string& signal, float value) { stroke_parameters.set_material_roughness(value); });
        gui.bind("pbr_metallic", [&](const std::string& signal, float value) { stroke_parameters.set_material_metallic(value); });

        gui.bind("color_picker", [&](const std::string& signal, Color color) { stroke_parameters.set_material_color(color); });

        // Controller buttons

        helper_gui.bind(XR_BUTTON_B, [&]() { stamp_enabled = !stamp_enabled; });

        // Bind recent color buttons...

        ui::UIEntity* recent_group = gui.get_widget_from_name("g_recent_colors");
        if (!recent_group) {
            assert(0);
            spdlog::error("Cannot find recent_colors button group!");
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
                    stroke_parameters.set_material_color(color);
                });
            }
        }

        max_recent_colors = recent_group->get_children().size();
        for (size_t i = 0; i < max_recent_colors; ++i)
        {
            ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(recent_group->get_children()[i]);
            gui.bind(child->signal, [&](const std::string& signal, void* button) {
                stroke_parameters.set_material_color((static_cast<ui::ButtonWidget*>(button))->color);
            });
        }

        // Bind material samples callback...

        ui::UIEntity* samples_group = gui.get_widget_from_name("g_material_samples");
        for (size_t i = 0; i < samples_group->get_children().size(); ++i)
        {
            ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(samples_group->get_children()[i]);
            gui.bind(child->signal, [&](const std::string& signal, void* button) {
                const PBRMaterialData& data = pbr_materials_data[signal];
                // Set all data
                stroke_parameters.set_material_color(data.base_color);
                stroke_parameters.set_material_roughness(data.roughness * 1.5f); // this is a hack because hdres don't have too much roughness..
                stroke_parameters.set_material_metallic(data.metallic);
                // Emit signals to change UI values
                gui.emit_signal("pbr_roughness@changed", stroke_parameters.get_material().roughness);
                gui.emit_signal("pbr_metallic@changed", stroke_parameters.get_material().metallic);
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
}

void SculptEditor::add_recent_color(const Color& color)
{
    auto it = std::find(recent_colors.begin(), recent_colors.end(), color);

    // Color is already in recents...
    if (it != recent_colors.end())
    {
        recent_colors.erase(it);
    }

    // Always add at the beginning
    recent_colors.insert(recent_colors.begin(), color);

    if (recent_colors.size() > max_recent_colors)
    {
        recent_colors.pop_back();
    }

    ui::UIEntity* recent_group = gui.get_widget_from_name("g_recent_colors");

    assert(recent_colors.size() <= recent_group->get_children().size());
    for (uint8_t i = 0; i < recent_colors.size(); ++i)
    {

        ui::ButtonWidget* child = static_cast<ui::ButtonWidget*>(recent_group->get_children()[i]);
        child->color = recent_colors[i];
        child->set_surface_material_override_color(0, child->color);
    }
}

void SculptEditor::add_pbr_material_data(const std::string& name, const Color& base_color, float roughness, float metallic)
{
    pbr_materials_data[name] = { .base_color = base_color, .roughness = roughness, .metallic = metallic };
}
