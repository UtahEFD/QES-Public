#pragma once

/*
 * This is an abstract class that is a child class of Building. The
 * non-polygonal building has additional information that is shared
 * by all buildings but polygon based buildings. This serves as a 
 * commonality.
 */

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