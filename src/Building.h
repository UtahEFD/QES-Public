#pragma once

#include "ParseInterface.h"

/*
 *Placeholder class for parsed building info in the xml
 */
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
	float rotation = 0;
	float Lf, Lr, Weff, Leff, Wt, Lt;
	int iStart, iEnd, jStart, jEnd, kStart, kEnd;
	int buildingRoof = 0;

	virtual void parseValues() = 0;
};