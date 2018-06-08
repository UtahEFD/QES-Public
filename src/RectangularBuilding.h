#pragma once

#include "ParseInterface.h"
#include "Building.h"

class RectangularBuilding : public Building
{
private:
	int height;
	int baseHeight;
	int centroidX;
	int centroidY;
	int xFo;
	int yFo;
	int length;
	int width;
	int rotation;

public:

	virtual void parseValues()
	{
		parsePrimative<int>(true, groupID, "groupID");
		parsePrimative<int>(true, buildingType, "buildingType");
		parsePrimative<int>(true, height, "height");
		parsePrimative<int>(true, baseHeight, "baseHeight");
		parsePrimative<int>(true, centroidX, "centroidX");
		parsePrimative<int>(true, centroidY, "centroidY");
		parsePrimative<int>(true, xFo, "xFo");
		parsePrimative<int>(true, yFo, "yFo");
		parsePrimative<int>(true, length, "length");
		parsePrimative<int>(true, width, "width");
		parsePrimative<int>(true, rotation, "rotation");
	}
};