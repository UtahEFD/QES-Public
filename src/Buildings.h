#pragma once

#include "ParseInterface.h"
#include "Building.h"
#include "RectangularBuilding.h"
/*
 *Placeholder class for parsed buildings info in the xml
 */
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
		parsePrimitive<int>(true, numBuildings, "numBuildings");
		parsePrimitive<int>(true, numPolygonNodes, "numPolygonNodes");
		parseMultiPolymorphs(true, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));
		parsePrimitive<float>(true, wallRoughness, "wallRoughness");

	}
};