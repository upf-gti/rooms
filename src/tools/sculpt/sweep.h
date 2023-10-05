#pragma once

#include "tool.h"

#define STARTING_ARC_LENGTH 100u
#define ARC_LENGTH_INCREASE 50u

class RoomsRenderer;

enum eSweepStrokeState : uint8_t {
    NO_STROKE = 0,
    IN_STROKE,
    FINISHED_STROKE
};

class SweepTool : public Tool {

    RoomsRenderer* renderer = nullptr;

    // Stroke state
    eSweepStrokeState stroke_prev_state = NO_STROKE;
    eSweepStrokeState stroke_state = NO_STROKE;

    // Stroke data
    glm::vec3   stroke_start_position = { 0.0f, 0.0f, 0.0f };
    glm::quat   stroke_start_orientation = { 0.0f, 0.0f, 0.0f, 1.0f };

    float inter_edit_distance = 0.01f;


    std::vector<Edit>   tmp_edit_storage;

    // Curve data
    float   *arc_length_LUT = nullptr;
    uint32_t arc_length_LUT_storeage_size = 0u;
    uint32_t arc_length_LUT_size = 0u;

    struct sParametricParabola {
        float curve_height = 0.0f;
        float curve_segment_size_pow_2 = 0.0f;

        float x(const float t) const;
        float y(const float t) const;

        inline void set_parabola(const float height, const float segment_size) {
            curve_height = height;
            curve_segment_size_pow_2 = segment_size * segment_size;
        }
    } curve;


    void fill_arc_length_LUT(const uint32_t element_count);
    uint32_t get_closest_arc_length(const float length) const;
    float aprox_inverse_curve_length(const float length) const;
    void fill_edits_with_arc();

    uint32_t get_number_of_edits_in_stroke(const glm::vec3& stroke_end_position, const glm::quat& stroke_end_orientation) const;

    void fill_edits_with_stroke();
public:

	void initialize() override;
	void clean() override;

	bool update(float delta_time) override;
	void render_scene() override;
	void render_ui() override;

	virtual bool use_tool() override;
};
