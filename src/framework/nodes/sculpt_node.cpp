#include "sculpt_node.h"

#include "framework/math/intersections.h"
#include "framework/resources/sculpt.h"
#include "framework/parsers/parse_scene.h"
#include "framework/nodes/node_factory.h"
#include "framework/nodes/mesh_instance_3d.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"
#include "graphics/managers/sculpt_manager.h"

#include "tools/sculpt_editor.h"
#include "tools/scene_editor.h"

#include "shaders/AABB_shader.wgsl.gen.h"

#include <engine/rooms_engine.h>

#include <fstream>

REGISTER_NODE_CLASS(SculptNode)

// #define SHOW_SCULPT_AABB

Stroke SculptNode::default_stroke = {
    .stroke_id = 0u,
    .edit_count = 1u,
    .primitive = SD_BOX,
    .operation = OP_SMOOTH_UNION,
    .parameters = { 0.f, -1.f, 0.f, 0.f },
    .edits = {
        { .position = {0.0f, 0.0f, 0.0f}, .dimensions = { 0.02f, 0.02f, 0.02f, 0.0f } }
    }
};

SculptNode::SculptNode() : Node3D()
{
    node_type = "SculptNode";
    collider_shape = COLLIDER_SHAPE_CUSTOM;

#ifdef SHOW_SCULPT_AABB
    AABB_mesh = parse_mesh("data/meshes/cube/aabb_cube.obj");

    Material* AABB_material = new Material();
    //AABB_material.priority = 10;
    AABB_material->set_color(glm::vec4(0.8f, 0.3f, 0.9f, 1.0f));
    AABB_material->set_transparency_type(ALPHA_BLEND);
    AABB_material->set_cull_type(CULL_NONE);
    AABB_material->set_type(MATERIAL_UNLIT);
    AABB_material->set_shader(RendererStorage::get_shader_from_source(shaders::AABB_shader::source, shaders::AABB_shader::path, AABB_material));
    //AABB_material.diffuse_texture = RendererStorage::get_texture("data/meshes/cube/cube_AABB.png");
    AABB_mesh->set_surface_material_override(AABB_mesh->get_surface(0), AABB_material);
#endif
}

SculptNode::SculptNode(SculptNode* reference) : SculptNode()
{
    sculpt_gpu_data = reference->get_sculpt_data();
    sculpt_gpu_data->ref();

    from_memory = true;
}

SculptNode::~SculptNode()
{
    // Remove from raymarching renderer

    if (sculpt_gpu_data->unref()) {
        static_cast<RoomsRenderer*>(Renderer::instance)->get_sculpt_manager()->delete_sculpt(sculpt_gpu_data);
    }
}

void SculptNode::initialize()
{
    // Create default sculpt
    std::vector<Stroke> history;
    history.push_back(default_stroke);
    from_history(history, false);
}

void SculptNode::update(float delta_time)
{
    uint32_t flags = 0u;
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(Renderer::instance);
    RoomsEngine* engine = static_cast<RoomsEngine*>(Engine::instance);
    auto scene_editor = engine->get_editor<SceneEditor*>(SCENE_EDITOR);
    sGPU_RayIntersectionData& intersection_results = renderer->get_sculpt_manager()->read_results.loaded_results.ray_intersection;

    bool in_sculpt_editor = (engine->get_current_editor_type() == SCULPT_EDITOR);
    bool in_scene_editor = (engine->get_current_editor() == scene_editor);
    bool editing_scene_group = in_scene_editor && (!!scene_editor->get_current_group());

    bool oof = false;

    if (in_sculpt_editor) {
        oof |= engine->get_editor<SculptEditor*>(SCULPT_EDITOR)->is_out_of_focus(this);
    }
    else if (editing_scene_group) {
        oof |= (!parent || parent != (Node*)scene_editor->get_current_group());
    }

    if (oof) {
        flags |= SCULPT_IS_OUT_OF_FOCUS;
    }

    /* Do not highlight if:
    * 1) Out of focus
    * 2) In sculpt mode
    * 3) No intersection
    */
    if (!oof && !in_sculpt_editor) {

        // check its intersection and its sibling ones if not in group editor
        bool hovered = check_intersection(&intersection_results);
        bool selected = (scene_editor->get_selected_node() == this);

        if (!editing_scene_group && parent) {

            selected |= (scene_editor->get_selected_node() == parent);

            for (auto child : parent->get_children()) {
                SculptNode* sculpt_child = dynamic_cast<SculptNode*>(child);
                hovered |= (sculpt_child && sculpt_child->check_intersection(&intersection_results));
            }
        }

        if (hovered) {
            flags |= SCULPT_IS_HOVERED;
        }
        else if (selected) {
            flags |= SCULPT_IS_SELECTED;
        }
    }
    
    in_frame_sculpt_render_list_id = renderer->add_sculpt_render_call(sculpt_gpu_data, get_global_model(), flags);

    // static_cast<RoomsRenderer*>(Renderer::instance)->add_sculpt_render_call(sculpt_gpu_data, glm::translate(get_global_model(), {0.05, 0.0, 0.0}));

    Node3D::update(delta_time);
}

void SculptNode::render()
{
#ifdef SHOW_SCULPT_AABB
    const AABB& sculpt_aabb = sculpt_gpu_data->get_AABB();
    AABB_mesh->set_scale(sculpt_aabb.half_size*2.0f);
    AABB_mesh->set_position(get_global_model() * glm::vec4(sculpt_aabb.center, 1.0));
    AABB_mesh->render();
#endif
}

void SculptNode::from_history(const std::vector<Stroke>& new_history, bool loaded_from_memory)
{
    if (!new_history.empty()) {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(Renderer::instance);
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt_from_history(new_history);
        sculpt_gpu_data->ref();
        from_memory = loaded_from_memory;
    }
    else {
        initialize();
    }
}

void SculptNode::serialize(std::ofstream& binary_scene_file)
{
    Node3D::serialize(binary_scene_file);

    sSculptBinaryHeader header = {
        .stroke_count = sculpt_gpu_data->get_stroke_history().size(),
    };

    binary_scene_file.write(reinterpret_cast<char*>(&header), sizeof(sSculptBinaryHeader));

    for (auto& stroke : sculpt_gpu_data->get_stroke_history()) {
        binary_scene_file.write(reinterpret_cast<char*>(&stroke), sizeof(Stroke));
    }
}

void SculptNode::parse(std::ifstream& binary_scene_file)
{
    Node3D::parse(binary_scene_file);

    sSculptBinaryHeader header;
    binary_scene_file.read(reinterpret_cast<char*>(&header), sizeof(sSculptBinaryHeader));

    RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(Renderer::instance);

    if (header.stroke_count > 0u) {
        std::vector<Stroke> stroke_history;
        stroke_history.resize(header.stroke_count);
        binary_scene_file.read(reinterpret_cast<char*>(&stroke_history[0]), header.stroke_count * sizeof(Stroke));
        from_history(stroke_history);
    } else {
        initialize();
    }


    // TODO: Remove current
    //rooms_renderer->get_raymarching_renderer()->set_current_sculpt(this);
    rooms_renderer->toogle_frame_debug();
}

void SculptNode::clone(Node* new_node, bool copy)
{
    Node3D::clone(new_node, copy);

    SculptNode* new_sculpt = static_cast<SculptNode*>(new_node);
    new_sculpt->set_from_memory(true);

    // instance copy, it should have different model, but uses same gpu data
    if (!copy) {
        new_sculpt->set_sculpt_data(sculpt_gpu_data);
        sculpt_gpu_data->ref();
    }
    // CLONE_COPY
    // raw copy, everything is recreated
    else {
        new_sculpt->from_history(get_sculpt_data()->get_stroke_history());
    }
}

bool SculptNode::check_intersection(sGPU_RayIntersectionData* data)
{
    return data->has_intersected == 1u && (sculpt_gpu_data->get_sculpt_id() == data->sculpt_id) && (in_frame_sculpt_render_list_id == data->instance_id);
}

bool SculptNode::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance, Node3D** out)
{
    const AABB& aabb = sculpt_gpu_data->get_AABB();

    glm::vec3 center = aabb.center + transform.get_position();

    const bool intersecting = intersection::ray_AABB(ray_origin, ray_direction, center, aabb.half_size, distance);

    if (intersecting) {
        sculpt_flags |= SCULPT_IS_HOVERED;
    } else {
        sculpt_flags &= ~SCULPT_IS_HOVERED;
    }

    return intersecting;
}


void SculptNode::set_out_of_focus(const bool oof)
{
    if (oof) {
        sculpt_flags |= eSculptInstanceFlags::SCULPT_IS_OUT_OF_FOCUS;
    }
    else {
        sculpt_flags &= ~eSculptInstanceFlags::SCULPT_IS_OUT_OF_FOCUS;
    }
}


uint32_t SculptNode::get_in_frame_model_idx()
{
    return in_frame_sculpt_render_list_id + sculpt_gpu_data->get_in_frame_model_buffer_index();
}
