#include "sculpt_node.h"

#include "framework/math/intersections.h"
#include "framework/resources/sculpt.h"
#include "framework/nodes/node_factory.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/managers/sculpt_manager.h"

#include "tools/sculpt_editor.h"
#include "tools/scene_editor.h"

#include <engine/rooms_engine.h>

#include <fstream>

REGISTER_NODE_CLASS(SculptNode)

SculptNode::SculptNode() : Node3D()
{
    node_type = "SculptNode";
    collider_shape = COLLIDER_SHAPE_CUSTOM;
}

SculptNode::SculptNode(SculptNode* reference) : Node3D()
{
    node_type = "SculptNode";
    collider_shape = COLLIDER_SHAPE_CUSTOM;

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

void SculptNode::update(float delta_time)
{
    uint32_t flags = 0u;
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(Renderer::instance);
    RoomsEngine* engine = static_cast<RoomsEngine*>(Engine::instance);
    const sGPU_SculptResults::sGPU_IntersectionData& intersection_results = renderer->get_sculpt_manager()->read_results.loaded_results.ray_intersection;
    bool in_sculpt_editor = (engine->get_current_editor_type() == SCULPT_EDITOR);
    bool editing_scene_group = (engine->get_current_editor_type() == SCENE_EDITOR) && (!!engine->get_editor<SceneEditor*>(SCENE_EDITOR)->get_current_group());

    bool oof = false;

    if (in_sculpt_editor) {
        oof |= (engine->get_editor<SculptEditor*>(SCULPT_EDITOR)->get_current_sculpt() != this);
    }
    else if (editing_scene_group) {
        oof |= (!parent || parent != (Node*)engine->get_editor<SceneEditor*>(SCENE_EDITOR)->get_current_group());
    }

    if (oof) {
        flags |= SCULPT_IS_OUT_OF_FOCUS;
    }

    /* Do not highlight if:
    * 1) Out of focus
    * 2) In sculpt mode
    * 3) No intersection
    */
    if (!oof && !in_sculpt_editor && intersection_results.has_intersected == 1u) {

        // check its intersection and its sibling ones if not in group editor
        bool hovered = check_intersection(intersection_results.sculpt_id, intersection_results.instance_id);

        if (!editing_scene_group && parent) {
            for (auto child : parent->get_children()) {
                SculptNode* sculpt_child = dynamic_cast<SculptNode*>(child);
                hovered |= (sculpt_child && sculpt_child->check_intersection(intersection_results.sculpt_id, intersection_results.instance_id));
            }
        }

        if (hovered) {
            flags |= SCULPT_IS_POINTED;
        }
    }
    
    in_frame_sculpt_render_list_id = renderer->add_sculpt_render_call(sculpt_gpu_data, get_global_model(), flags);

    // static_cast<RoomsRenderer*>(Renderer::instance)->add_sculpt_render_call(sculpt_gpu_data, glm::translate(get_global_model(), {0.05, 0.0, 0.0}));

    Node3D::update(delta_time);
}

void SculptNode::render()
{
    
}

void SculptNode::initialize()
{
    sculpt_gpu_data = static_cast<RoomsRenderer*>(Renderer::instance)->get_sculpt_manager()->create_sculpt();
    sculpt_gpu_data->ref();
}

void SculptNode::from_history(const std::vector<Stroke>& new_history)
{
    if (!new_history.empty()) {
        RoomsRenderer* rooms_renderer = static_cast<RoomsRenderer*>(Renderer::instance);
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt_from_history(new_history);
        sculpt_gpu_data->ref();
        from_memory = true;
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
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt();
        sculpt_gpu_data->ref();
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

bool SculptNode::check_intersection(uint32_t sculpt_id, uint32_t instance_id)
{
    return (sculpt_gpu_data->get_sculpt_id() == sculpt_id) && (in_frame_sculpt_render_list_id == instance_id);
}

bool SculptNode::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance)
{
    // wip... maybe this instance "aabb" is correct and we don't have to compute this every time

    AABB current_aabb;

    for (const auto& stroke : sculpt_gpu_data->get_stroke_history()) {
        current_aabb = merge_aabbs(current_aabb, stroke.get_world_AABB());
    }

    glm::vec3 center = current_aabb.center + transform.get_position();

    const bool intersecting = intersection::ray_AABB(ray_origin, ray_direction, center, current_aabb.half_size, distance);

    if (intersecting) {
        sculpt_flags |= SCULPT_IS_POINTED;
    } else {
        sculpt_flags &= ~SCULPT_IS_POINTED;
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
