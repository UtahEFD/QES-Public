#pragma once

#include "Building.h"


class NonPolyBuilding : public Building
{
private:


public:
	float xFo;
	float yFo;
	float length;
	float width;

	virtual void parseValues() = 0;
};