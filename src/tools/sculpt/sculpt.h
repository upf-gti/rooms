#pragma once

#include "tool.h"

class SculptTool : public Tool {

    EntityMesh* mesh_preview = nullptr;
    ui::Controller ui_controller;

public:

	void initialize();
	void clean();

	bool update(float delta_time);
	void render_scene();
	void render_ui();

	virtual bool use_tool() override;
};
