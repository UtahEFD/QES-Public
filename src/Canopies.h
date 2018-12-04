#pragma once

#include "util/ParseInterface.h"
#include "Building.h"
#include "Canopy.h"

class Canopies : public ParseInterface
{
private:



public:

	int num_canopies;
	int landuse_flag;
	int landuse_veg_flag;
	int landuse_urb_flag;
	std::vector<Building*> canopies;


	virtual void parseValues()
	{
		parsePrimitive<int>(true, num_canopies, "num_canopies");
		parsePrimitive<int>(true, landuse_flag, "landuseFlag");
		parsePrimitive<int>(true, landuse_veg_flag, "landuseVegetationFlag");
		parsePrimitive<int>(true, landuse_urb_flag, "landuseUrbanFlag");
		parseMultiPolymorphs(true, canopies, Polymorph<Building, Canopy>("canopy"));

	}
};
