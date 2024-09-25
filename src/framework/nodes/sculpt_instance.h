#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"
#include "graphics/pipeline.h"

#include <vector>

struct GPUSculptData {
    uint32_t octree_id;
    Uniform octree_uniform;
    WGPUBindGroup octree_bindgroup = nullptr;
};

class SculptInstance : public Node3D {

    std::vector<Stroke> stroke_history;

    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

    GPUSculptData sculpt_gpu_data;

public:

    SculptInstance();
    SculptInstance(SculptInstance *reference);
    ~SculptInstance();

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

    void initialize();

    void from_history(const std::vector<Stroke>& new_history);

    virtual void serialize(std::ofstream& binary_scene_file);
    virtual void parse(std::ifstream& binary_scene_file);

    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance) override;
};
