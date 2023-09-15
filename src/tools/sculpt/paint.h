#pragma once

#include "tool.h"

class PaintTool : public Tool {

public:

	void initialize();
	void clean();

    bool update(float delta_time);
	void render_scene();
	void render_ui();

	virtual bool use_tool() override;
};
