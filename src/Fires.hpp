#pragma once

#include "util/ParseInterface.h"

class Fires : public ParseInterface
{
private:

public:

	int numFires,fuelType;
	float height,baseHeight,xStart,yStart,length,width;

	virtual void parseValues()
	{
		parsePrimitive<int>(true,   numFires,   "numFires");
		parsePrimitive<int>(true,   fuelType,   "fuelType");
		parsePrimitive<float>(true, height,     "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, xStart,     "xStart");
		parsePrimitive<float>(true, yStart,     "yStart");
		parsePrimitive<float>(true, length,     "length");
		parsePrimitive<float>(true, width,      "width");
	}
};