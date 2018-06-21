#pragma once

#include "ParseInterface.h"

class Building : public ParseInterface
{
protected:

public:
	int groupID;
	int buildingType, buildingGeometry;
	float height;
	float baseHeight, baseHeightActual; //zfo
	float centroidX;
	float centroidY;
	int buildingDamage = 0;
	float atten = 0;

	virtual void parseValues() = 0;
};