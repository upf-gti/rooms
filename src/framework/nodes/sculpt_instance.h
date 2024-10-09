#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"
#include "graphics/pipeline.h"

#include <vector>

class Sculpt;

class SculptInstance : public Node3D {
    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

    uint32_t sculpt_flags = 0u;
    bool are_sculpt_flags_dirty = false;

    Sculpt* sculpt_gpu_data;

public: 

    SculptInstance();
    SculptInstance(SculptInstance *reference);
    ~SculptInstance();

    void update(float delta_time);

    void render() override;

    inline void set_sculpt_data(Sculpt* new_data) {
        sculpt_gpu_data = new_data;
    }

    inline Sculpt* get_sculpt_data() {
        return sculpt_gpu_data;
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
