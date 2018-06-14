#pragma once

#include "ParseInterface.h"

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
		parsePrimative<int>(true, siteCoords, "siteCoords");
		parsePrimative<int>(true, x, "x");
		parsePrimative<int>(true, y, "y");
		parsePrimative<float>(true, epoch, "epoch");
		parsePrimative<bool>(true, boundaryLayerFlag, "boundaryLayerFlag");
		parsePrimative<float>(true, siteZ0, "siteZ0");
		parsePrimative<int>(true, reciprocal, "reciprocal");
		parsePrimative<int>(true, height, "height");
		parsePrimative<int>(true, speed, "speed");
		parsePrimative<int>(true, direction, "direction");

	}
};