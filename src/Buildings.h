#pragma once

#include "ParseInterface.h"
#include "Building.h"
#include "RectangularBuilding.h"

class Buildings : public ParseInterface
{
private:



public:

	float wallRoughness;
	int numBuildings;
	int numPolygonNodes;
	std::vector<Building*> buildings;

	virtual void parseValues()
	{
		parsePrimative<float>(true, wallRoughness, "wallRoughness");
		parsePrimative<int>(true, numBuildings, "numBuildings");
		parsePrimative<int>(true, numPolygonNodes, "numPolygonNodes");
		parseMultiPolymorphs(true, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));

	}
};