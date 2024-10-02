#pragma once

#include "graphics/pipeline.h"
#include "framework/resources/resource.h"
#include "graphics/edit.h"


#include <glm/glm.hpp>
#include <vector>

class Sculpt : public Resource {

    uint32_t sculpt_id;

    Uniform octree_uniform;
    WGPUBindGroup octree_bindgroup = nullptr;

    std::vector<Stroke> stroke_history;

public:
    Sculpt(const uint32_t id, const Uniform uniform, const WGPUBindGroup bindgroup):
        sculpt_id (id), octree_uniform(uniform), octree_bindgroup(bindgroup) {};

    void init();
    void clean();

    uint32_t get_sculpt_id() { return sculpt_id; }

    void set_octree_uniform(Uniform octree_uniform) { this->octree_uniform = octree_uniform; }
    void set_octree_bindgroup(WGPUBindGroup octree_bindgroup) { this->octree_bindgroup = octree_bindgroup; }
    void set_stroke_history(const std::vector<Stroke>& stroke_history) { this->stroke_history = stroke_history; }

    Uniform get_octree_uniform() { return octree_uniform; }
    WGPUBindGroup get_octree_bindgroup() { return octree_bindgroup; }
    std::vector<Stroke>& get_stroke_history() { return stroke_history; }

};
