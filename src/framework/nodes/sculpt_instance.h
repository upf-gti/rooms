#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"

#include <vector>

class SculptInstance : public Node3D {

    std::vector<Stroke> stroke_history;

    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

public:

    SculptInstance();
    ~SculptInstance();

    std::vector<Stroke>& get_stroke_history();

    virtual void serialize(std::ofstream& binary_scene_file);
    virtual void parse(std::ifstream& binary_scene_file);

};
