#pragma once

#include "includes.h"

class Tool {

protected:

	bool selected = false;

public:

	Tool() {};

	/*
	*	Events
	*/

	virtual void on_press() {};
};