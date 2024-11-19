#pragma once

#include "graphics/pipeline.h"
#include "framework/resources/resource.h"
#include "graphics/edit.h"


#include <glm/glm.hpp>
#include <vector>

class Sculpt : public Resource {
    uint32_t in_frame_model_buffer_index = 0u;
    uint32_t sculpt_id;

    Uniform octree_uniform;

    // For calling when rendering the scupt
    Uniform indirect_call_buffer;

    Uniform brick_indices_buffer;

    WGPUBindGroup octree_bindgroup = nullptr;
    WGPUBindGroup evaluate_sculpt_bindgroup = nullptr;
    WGPUBindGroup readonly_octree_bindgroup = nullptr;
    WGPUBindGroup octree_indirect_bindgroup = nullptr;
    WGPUBindGroup octree_brick_copy_aabb_gen_bindgroup = nullptr;

    AABB aabb;

    std::vector<Stroke> stroke_history;

    bool deleted = false;

    void on_delete() override;

public:
    Sculpt(const uint32_t id, const Uniform uniform, const Uniform indiret_buffer, const Uniform indices_buffer, const WGPUBindGroup octree_bind):
        sculpt_id (id), octree_uniform(uniform), octree_bindgroup(octree_bind),
        indirect_call_buffer(indiret_buffer), brick_indices_buffer(indices_buffer) {};

    /*void init();
    void clean();*/

    void set_octree_uniform(Uniform octree_uniform) { this->octree_uniform = octree_uniform; }
    void set_octree_bindgroup(WGPUBindGroup octree_bindgroup) { this->octree_bindgroup = octree_bindgroup; }
    void set_stroke_history(const std::vector<Stroke>& stroke_history) { this->stroke_history = stroke_history; }
    void set_brick_copy_bindgroup(const WGPUBindGroup brick_copy_aabb_gen_bindgroup) { octree_brick_copy_aabb_gen_bindgroup = brick_copy_aabb_gen_bindgroup; }
    void set_readonly_octree_bindgroup(const WGPUBindGroup readonly_bindgroup) { readonly_octree_bindgroup = readonly_bindgroup; }
    void set_indirect_buffers_bindgroup(const WGPUBindGroup oct_indir_bindroup) { octree_indirect_bindgroup = oct_indir_bindroup; }
    void set_sculpt_evaluation_bindgroup(const WGPUBindGroup eval_sculpt_bindgroup) { evaluate_sculpt_bindgroup = eval_sculpt_bindgroup; }
    void mark_as_deleted() { deleted = true; }

    uint32_t get_sculpt_id() const { return sculpt_id; }
    Uniform &get_octree_uniform() { return octree_uniform; }
    WGPUBindGroup get_octree_bindgroup() { return octree_bindgroup; }
    WGPUBindGroup get_sculpt_bindgroup() { return evaluate_sculpt_bindgroup; }
    WGPUBindGroup get_readonly_sculpt_bindgroup() { return readonly_octree_bindgroup; }
    WGPUBindGroup get_octree_indirect_bindgroup() { return octree_indirect_bindgroup; }
    WGPUBindGroup get_brick_copy_aabb_gen_bindgroup() { return octree_brick_copy_aabb_gen_bindgroup; }
    std::vector<Stroke>& get_stroke_history() { return stroke_history; }
    Uniform& get_brick_indices_uniform() { return brick_indices_buffer; }
    Uniform& get_indirect_render_buffer() { return indirect_call_buffer; }

    inline uint32_t get_in_frame_model_buffer_index() { return in_frame_model_buffer_index; }
    inline void set_in_frame_model_buffer_index(const uint32_t id) { in_frame_model_buffer_index = id; }

    bool is_deleted() { return deleted; }

    const AABB& get_AABB() const { return aabb; }
    void set_AABB(const AABB& new_aabb) { aabb = new_aabb; }
};
