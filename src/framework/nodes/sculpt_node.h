#pragma once

#include "framework/nodes/node_3d.h"
#include "graphics/edit.h"
#include "graphics/pipeline.h"

#include <vector>

class Sculpt;

class SculptNode : public Node3D {

    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

    uint32_t sculpt_flags = 0u;
    bool are_sculpt_flags_dirty = false;

    Sculpt* sculpt_gpu_data = nullptr;

    uint32_t in_frame_instance_id = 0u;

    void from_history(const std::vector<Stroke>& new_history);

public: 

    SculptNode();
    SculptNode(SculptNode* reference);
    ~SculptNode();

    void initialize();

    void update(float delta_time);
    void render() override;

    virtual void serialize(std::ofstream& binary_scene_file);
    virtual void parse(std::ifstream& binary_scene_file);

    void clone(Node* new_node, bool copy = true) override;
    bool test_ray_collision(const glm::vec3& ray_origin, const glm::vec3& ray_direction, float& distance) override;

    inline void set_sculpt_data(Sculpt* new_data) { sculpt_gpu_data = new_data; }
    void set_out_of_focus(const bool oof);

    inline Sculpt* get_sculpt_data() { return sculpt_gpu_data; }
    inline uint32_t get_flags() { return sculpt_flags; }
    inline uint32_t get_in_frame_render_instance_idx() const { return in_frame_instance_id; }
};