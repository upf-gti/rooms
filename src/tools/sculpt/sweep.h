#pragma once

#include "tool.h"

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
