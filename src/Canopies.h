#pragma once

#include "util/ParseInterface.h"
#include "Building.h"

#include "Canopy.h"
#include "CanopyHomogeneous.h"
#include "CanopyIsolatedTree.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


class Canopies : public ParseInterface
{
private:
    
    
    
public:
    
    int num_canopies;
    std::vector<Building*> canopies;
    
    
    virtual void parseValues()
    {
        parsePrimitive<int>(true, num_canopies, "num_canopies");
        // read the input data for canopies
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("CanopyHomogeneous"));
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("CanopyIsolatedTree"));
        // add other type of canopy here
    }
};
