#pragma once

#include "ParseInterface.h"
#include "Building.h"
#include "RectangularBuilding.h"

class Buildings : public ParseInterface
{
private:



public:

	int numBuildings;
	int numPolygonNodes;
	std::vector<Building*> buildings;
	float wallRoughness;

	virtual void parseValues()
	{
		parsePrimative<int>(true, numBuildings, "numBuildings");
		parsePrimative<int>(true, numPolygonNodes, "numPolygonNodes");
		parseMultiPolymorphs(true, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));
		parsePrimative<float>(true, wallRoughness, "wallRoughness");

	}
};