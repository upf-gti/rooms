#pragma once

#include "tool.h"

#define STARTING_ARC_LENGTH 100u
#define ARC_LENGTH_INCREASE 50u

class RoomsRenderer;
class SculptEditor;

enum eSweepStrokeState : uint8_t {
    NO_STROKE = 0,
    IN_STROKE,
    FINISHED_STROKE
};

class SweepTool : public Tool {
    SculptEditor*           sculpt_editor = nullptr;

    // Stroke state
    eSweepStrokeState       stroke_prev_state = NO_STROKE;
    eSweepStrokeState       stroke_state = NO_STROKE;

    // Stroke data
    glm::vec3               stroke_start_position = { 0.0f, 0.0f, 0.0f };
    glm::quat               stroke_start_orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    float                   inter_edit_distance = 0.01f;

    // Edit data
    std::vector<Edit>       tmp_edit_storage;
    glm::vec4               start_edit_dimensions;

    // Curve data
    float                   *arc_length_LUT = nullptr;
    uint32_t                arc_length_LUT_storeage_size = 0u;
    uint32_t                arc_length_LUT_size = 0u;
    float                   curve_length = 0.0f;

    struct sBezierCurve {
        glm::vec3       start;
        glm::vec3       control;
        glm::vec3       end;

        inline void set_curve(const glm::vec3& s,
                              const glm::vec3& c,
                              const glm::vec3& e) {
            start = s;
            control = c;
            end = e;
        }

        glm::vec3 evaluate(const float t) const;
    } curve;


    // Curve sweep and stroke functions
    void fill_arc_length_LUT(const uint32_t element_count, const float bowstring_length);
    uint32_t get_closest_arc_length(const float length) const;
    float aprox_inverse_curve_length(const float length) const;
    void fill_edits_with_arc(const float delta);
    uint32_t get_number_of_edits_in_stroke(const glm::vec3& stroke_end_position, const glm::quat& stroke_end_orientation) const;
    void fill_edits_with_stroke();
public:

	void initialize() override;
	void clean() override;

    void set_sculpt_editor(SculptEditor* editor) {
        sculpt_editor = editor;
    }

	bool update(float delta_time) override;
	void render_scene() override;
	void render_ui() override;

	virtual bool use_tool() override;
};
