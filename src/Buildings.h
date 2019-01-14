#pragma once

/*
 * This class contains data and variables that pretain to all buildings
 * along with a list of all buildings pulled from an input xml file
 */

#include "util/ParseInterface.h"
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
		parsePrimitive<int>(true, numBuildings, "numBuildings");
		parsePrimitive<int>(true, numPolygonNodes, "numPolygonNodes");
		parseMultiPolymorphs(true, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));
		parsePrimitive<float>(true, wallRoughness, "wallRoughness");



	}
};
