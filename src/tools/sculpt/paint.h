#pragma once

#include "tool.h"

class PaintTool : public Tool {

public:

	void initialize() override;
	void clean() override;

    bool update(float delta_time) override;
	void render_scene() override;
	void render_ui() override;

	virtual bool use_tool() override;
};
