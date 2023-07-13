#pragma once

#include "engine.h"

class RoomsEngine : public Engine {

	std::vector<Entity*> entities;

public:

	virtual int initialize(Renderer* renderer, GLFWwindow* window, bool use_mirror_screen) override;

	virtual void update(float delta_time) override;
	virtual void render() override;
};
