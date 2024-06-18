#include "sculpt_instance.h"
#include "graphics/renderers/rooms_renderer.h"

#include <fstream>

SculptInstance::SculptInstance() : Node3D()
{
    node_type = "SculptInstance";

    SculptureData new_sculpt_data = dynamic_cast<RoomsRenderer*>(Renderer::instance)->get_raymarching_renderer()->create_new_sculpture();

    sculpture_octree_uniform = new_sculpt_data.sculpture_octree_uniform;
    sculpture_octree_bindgroup = new_sculpt_data.sculpture_octree_bindgroup;
    octree_id = new_sculpt_data.octree_id;
}

SculptInstance::~SculptInstance()
{
}

std::vector<Stroke>& SculptInstance::get_stroke_history()
{
    return stroke_history;
}

void SculptInstance::serialize(std::ofstream& binary_scene_file)
{
    Node3D::serialize(binary_scene_file);

    sSculptBinaryHeader header = {
        .stroke_count = stroke_history.size(),
    };

    binary_scene_file.write(reinterpret_cast<char*>(&header), sizeof(sSculptBinaryHeader));

    for (auto& stroke : stroke_history) {
        binary_scene_file.write(reinterpret_cast<char*>(&stroke), sizeof(Stroke));
    }
}

void SculptInstance::parse(std::ifstream& binary_scene_file)
{
    Node3D::parse(binary_scene_file);

    sSculptBinaryHeader header;
    binary_scene_file.read(reinterpret_cast<char*>(&header), sizeof(sSculptBinaryHeader));

    stroke_history.resize(header.stroke_count);
    binary_scene_file.read(reinterpret_cast<char*>(&stroke_history[0]), header.stroke_count * sizeof(Stroke));

    RoomsRenderer* rooms_renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);
    rooms_renderer->get_raymarching_renderer()->set_current_sculpt(this);
    rooms_renderer->toogle_frame_debug();
}
