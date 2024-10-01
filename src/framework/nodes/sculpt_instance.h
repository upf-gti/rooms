#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"
#include "graphics/pipeline.h"

#include <vector>

struct GPUSculptData {
    uint32_t octree_id;
    Uniform octree_uniform;
    WGPUBindGroup octree_bindgroup = nullptr;

    void init();
    void clean();
};

class SculptInstance : public Node3D {
    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

    uint32_t sculpt_flags = 0u;
    bool are_sculpt_flags_dirty = false;

    GPUSculptData sculpt_gpu_data;

public: 

    SculptInstance();
    SculptInstance(SculptInstance *reference);
    ~SculptInstance();

    void update(float delta_time);

    std::vector<Stroke>& get_stroke_history();

    inline Uniform& get_octree_uniform() {
        return sculpt_gpu_data.octree_uniform;
    }

    inline WGPUBindGroup get_octree_bindgroup() const {
        return sculpt_gpu_data.octree_bindgroup;
    }

    inline void set_octree_uniform(const Uniform& uni) {
        sculpt_gpu_data.octree_uniform = uni;
    }

    inline void set_octree_bindgroup(const WGPUBindGroup bindgroup) {
        sculpt_gpu_data.octree_bindgroup = bindgroup;
    }

    inline uint32_t get_octree_id() const {
        return sculpt_gpu_data.octree_id;
    }

    inline void set_sculpt_data(const GPUSculptData& new_data) {
        sculpt_gpu_data = new_data;
    }

    inline uint32_t get_flags() {
        return sculpt_flags;
    }

    void set_out_of_focus(const bool oof);

    void initialize();

    void from_history(const std::vector<Stroke>& new_history);

    virtual void serialize(std::ofstream& binary_scene_file);
    virtual void parse(std::ifstream& binary_scene_file);

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance) override;
};
