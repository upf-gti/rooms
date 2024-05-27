#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"

#include <vector>

class SculptInstance : public Node3D {

    std::vector<Stroke> stroke_history;

public:

    SculptInstance();
    ~SculptInstance();

    std::vector<Stroke>& get_stroke_history();

};
