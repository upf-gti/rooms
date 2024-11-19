#pragma once

#include "framework/nodes/node_3d.h"

#include "graphics/edit.h"
#include "graphics/pipeline.h"

#include <vector>

class Sculpt;
class MeshInstance3D;
class sGPU_RayIntersectionData;

class SculptNode : public Node3D {

    struct sSculptBinaryHeader {
        size_t stroke_count = 0;
    };

    // bool are_sculpt_flags_dirty = false;
    bool from_memory = false;

    uint32_t sculpt_flags = 0u;
    uint32_t in_frame_sculpt_render_list_id = 0u;

    Sculpt* sculpt_gpu_data = nullptr;

    void from_history(const std::vector<Stroke>& new_history, bool loaded_from_memory = true);

    // DEBUG
    MeshInstance3D* AABB_mesh = nullptr;

    static Stroke default_stroke;

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
    bool check_intersection(sGPU_RayIntersectionData* data);

    inline void set_sculpt_data(Sculpt* new_data) { sculpt_gpu_data = new_data; }
    void set_out_of_focus(const bool oof);
    void set_from_memory(bool value) { from_memory = value; }

    inline Sculpt* get_sculpt_data() { return sculpt_gpu_data; }
    inline uint32_t get_flags() { return sculpt_flags; }
    inline uint32_t get_in_frame_render_instance_idx() const { return in_frame_sculpt_render_list_id; }
    inline bool get_from_memory() { return from_memory; }

    uint32_t get_in_frame_model_idx();
};
