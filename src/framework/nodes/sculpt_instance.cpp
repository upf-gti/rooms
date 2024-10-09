#include "sculpt_instance.h"

#include "framework/math/intersections.h"
#include "graphics/renderers/rooms_renderer.h"
#include "graphics/managers/sculpt_manager.h"

#include <fstream>

#include "framework/resources/sculpt.h"

SculptInstance::SculptInstance() : Node3D()
{
    node_type = "SculptInstance";
    collider_shape = COLLIDER_SHAPE_CUSTOM;
}

SculptInstance::SculptInstance(SculptInstance* reference) : Node3D()
{
    node_type = "SculptInstance";
    collider_shape = COLLIDER_SHAPE_CUSTOM;

    sculpt_gpu_data = reference->get_sculpt_data();
    sculpt_gpu_data->ref();

    //dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->add_sculpt_instance(this);
}

SculptInstance::~SculptInstance()
{
    // Remove from raymarching renderer

    //dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->remove_sculpt_instance(this);

    sculpt_gpu_data->unref();
}


void SculptInstance::update(float delta_time) {
    dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->add_rendercall_to_sculpt(sculpt_gpu_data, get_model());

    //dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->add_rendercall_to_sculpt(sculpt_gpu_data, glm::translate(glm::mat4(1.0f), {0.05, 0.0, 0.0}));
}

void SculptInstance::render() {
    
}

void SculptInstance::initialize()
{
    sculpt_gpu_data = dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_sculpt_manager()->create_sculpt();
    sculpt_gpu_data->ref();

    //dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->add_sculpt_instance(this);
}

void SculptInstance::from_history(const std::vector<Stroke>& new_history)
{
    if (!new_history.empty()) {
        RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt_from_history(new_history);
        sculpt_gpu_data->ref();
        //rooms_renderer->get_raymarching_renderer()->add_sculpt_instance(this);
    }
    else {
        initialize();
    }
}

void SculptInstance::serialize(std::ofstream& binary_scene_file)
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

void SculptInstance::parse(std::ifstream& binary_scene_file)
{
    Node3D::parse(binary_scene_file);

    sSculptBinaryHeader header;
    binary_scene_file.read(reinterpret_cast<char*>(&header), sizeof(sSculptBinaryHeader));

    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    if (header.stroke_count > 0u) {
        std::vector<Stroke> stroke_history;
        stroke_history.resize(header.stroke_count);
        binary_scene_file.read(reinterpret_cast<char*>(&stroke_history[0]), header.stroke_count * sizeof(Stroke));
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt_from_history(stroke_history);
    } else {
        sculpt_gpu_data = rooms_renderer->get_sculpt_manager()->create_sculpt();
    }

    sculpt_gpu_data->ref();

    //rooms_renderer->get_raymarching_renderer()->add_sculpt_instance(this);

    // TODO: Remove current
    //rooms_renderer->get_raymarching_renderer()->set_current_sculpt(this);
    rooms_renderer->toogle_frame_debug();
}

bool SculptInstance::test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance)
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


void SculptInstance::set_out_of_focus(const bool oof) {
    if (oof) {
        sculpt_flags |= eSculptInstanceFlags::SCULPT_IS_OUT_OF_FOCUS;
    }
    else {
        sculpt_flags &= ~eSculptInstanceFlags::SCULPT_IS_OUT_OF_FOCUS;
    }
}
