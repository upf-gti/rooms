#pragma once

#include "tooling/base.h"

class MirrorTool : public Tool {

public:

	MirrorTool() {};

	/*
	*	Events
	*/

	virtual void on_press() override;
};