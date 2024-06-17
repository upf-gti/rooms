#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"
#include "graphics/pipeline.h"


#include <vector>




class SculptInstance : public Node3D {

    std::vector<Stroke> stroke_history;
    Uniform sculpture_octree_uniform;
    WGPUBindGroup sculpture_octree_bindgroup = nullptr;

    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

public:

    SculptInstance();
    ~SculptInstance();

    std::vector<Stroke>& get_stroke_history();

    inline Uniform& get_octree_uniform() {
        return sculpture_octree_uniform;
    }

    inline WGPUBindGroup get_octree_bindgroup() {
        return sculpture_octree_bindgroup;
    }

    inline void set_octree_uniform(const Uniform &uni) {
        sculpture_octree_uniform = uni;
    }

    inline void set_octree_bindgroup(const WGPUBindGroup bindgroup) {
        sculpture_octree_bindgroup = bindgroup;
    }

    virtual void serialize(std::ofstream& binary_scene_file);
    virtual void parse(std::ifstream& binary_scene_file);

};
