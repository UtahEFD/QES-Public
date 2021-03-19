#pragma once

#include "util/ParseInterface.h"
#include "Building.h"

#include "Canopy.h"
#include "CanopyHomogeneous.h"
#include "CanopyIsolatedTree.h"
#include "CanopyWindbreak.h"
#include "GroundCover.h"
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


class Canopies : public ParseInterface
{
private:
    
    
    
public:
    
    int num_canopies;
    std::vector<Building*> canopies;
    std::vector<GroundCover*> groundCovers;
    
    virtual void parseValues()
    {
        parsePrimitive<int>(true, num_canopies, "num_canopies");
        // read the input data for canopies
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("Homogeneous"));
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("IsolatedTree"));
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyWindbreak>("Windbreak"));
        // add other type of canopy here
        
        parseMultiPolymorphs(false, groundCovers, Polymorph<GroundCover, GroundCoverRectangular>("GroundCoverRectangular"));
        
    }
};
