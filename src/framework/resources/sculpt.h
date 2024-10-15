#pragma once

#include "graphics/pipeline.h"
#include "framework/resources/resource.h"
#include "graphics/edit.h"


#include <glm/glm.hpp>
#include <vector>

class Sculpt : public Resource {

    uint32_t sculpt_id;

    Uniform octree_uniform;

    // For calling when rendering the scupt
    Uniform indirect_call_buffer;

    Uniform brick_indices_buffer;

    WGPUBindGroup octree_bindgroup = nullptr;
    WGPUBindGroup evaluate_sculpt_bindgroup = nullptr;
    WGPUBindGroup readonly_octree_bindgroup = nullptr;
    WGPUBindGroup octree_indirect_bindgroup = nullptr;

    std::vector<Stroke> stroke_history;

public:
    Sculpt(const uint32_t id, const Uniform uniform, const Uniform indiret_buffer, const Uniform indices_buffer, const WGPUBindGroup octree_bind, const WGPUBindGroup eval_sculpt_bindgroup, const WGPUBindGroup oct_indir_bindroup, const WGPUBindGroup readonly_bindgroup):
        sculpt_id (id), octree_uniform(uniform), evaluate_sculpt_bindgroup(eval_sculpt_bindgroup), octree_bindgroup(octree_bind),
        indirect_call_buffer(indiret_buffer), brick_indices_buffer(indices_buffer), readonly_octree_bindgroup(readonly_bindgroup),
        octree_indirect_bindgroup(oct_indir_bindroup) {};

    void init();
    void clean();

    uint32_t get_sculpt_id() const { return sculpt_id; }

    void set_octree_uniform(Uniform octree_uniform) { this->octree_uniform = octree_uniform; }
    void set_octree_bindgroup(WGPUBindGroup octree_bindgroup) { this->octree_bindgroup = octree_bindgroup; }
    void set_stroke_history(const std::vector<Stroke>& stroke_history) { this->stroke_history = stroke_history; }

    Uniform &get_octree_uniform() { return octree_uniform; }
    WGPUBindGroup get_octree_bindgroup() { return octree_bindgroup; }
    WGPUBindGroup get_sculpt_bindgroup() { return evaluate_sculpt_bindgroup; }
    WGPUBindGroup get_readonly_sculpt_bindgroup() { return readonly_octree_bindgroup; }
    WGPUBindGroup get_octree_indirect_bindgroup() { return octree_indirect_bindgroup; }

    std::vector<Stroke>& get_stroke_history() { return stroke_history; }

    Uniform &get_brick_indices_uniform() { return brick_indices_buffer; }

    Uniform& get_indirect_render_buffer() { return indirect_call_buffer; }
};
