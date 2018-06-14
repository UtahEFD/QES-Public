#pragma once

#include "ParseInterface.h"
#include "Building.h"

class RectangularBuilding : public Building
{
private:


public:
	float height;
	float baseHeight;
	float centroidX;
	float centroidY;
	float xFo;
	float yFo;
	float length;
	float width;
	float rotation;

	virtual void parseValues()
	{
		parsePrimative<int>(true, groupID, "groupID");
		parsePrimative<int>(true, buildingType, "buildingType");
		parsePrimative<float>(true, height, "height");
		parsePrimative<float>(true, baseHeight, "baseHeight");
		parsePrimative<float>(true, centroidX, "centroidX");
		parsePrimative<float>(true, centroidY, "centroidY");
		parsePrimative<float>(true, xFo, "xFo");
		parsePrimative<float>(true, yFo, "yFo");
		parsePrimative<float>(true, length, "length");
		parsePrimative<float>(true, width, "width");
		parsePrimative<float>(true, rotation, "rotation");
	}
};