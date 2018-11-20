#pragma once

#include "util/ParseInterface.h"

/*
 *Placeholder class for parsed sensor info in the xml
 */
class Sensor : public ParseInterface
{
private:

public:
		int siteCoords;
		int x, y;
		float epoch;
		bool boundaryLayerFlag;
		float siteZ0;
		int reciprocal;
		int height;
		int speed;
		int direction; 


	virtual void parseValues()
	{
		parsePrimitive<int>(true, siteCoords, "siteCoords");
		parsePrimitive<int>(true, x, "x");
		parsePrimitive<int>(true, y, "y");
		parsePrimitive<float>(true, epoch, "epoch");
		parsePrimitive<bool>(true, boundaryLayerFlag, "boundaryLayerFlag");
		parsePrimitive<float>(true, siteZ0, "siteZ0");
		parsePrimitive<int>(true, reciprocal, "reciprocal");
		parsePrimitive<int>(true, height, "height");
		parsePrimitive<int>(true, speed, "speed");
		parsePrimitive<int>(true, direction, "direction");

	}
};
