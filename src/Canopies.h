#pragma once

#include "util/ParseInterface.h"
#include "Building.h"

#include "Canopy.h"
#include "CanopyHomogeneous.h"
#include "CanopyIsolatedTree.h"
#include "CanopyWindbreak.h"

#include "ESRIShapefile.h"

#include "GroundCover.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


class Canopies : public ParseInterface
{
private:
    
    
    
public:
    
    int num_canopies;
    std::vector<Building*> canopies;
    
    // SHP File parameters
    std::string shpFile;   // SHP file name
    std::string shpTreeLayerName;
    ESRIShapefile *SHPData = nullptr;
    std::vector< std::vector <polyVert> > shpPolygons;
    std::vector <float> shpTreeHeight;  // Height of buildings
    
    std::vector<GroundCover*> groundCovers;
    
    virtual void parseValues()
    {
        parsePrimitive<int>(true, num_canopies, "num_canopies");
        // read the input data for canopies
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("Homogeneous"));
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("IsolatedTree"));
        parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyWindbreak>("Windbreak"));
        
        parseMultiPolymorphs(false, groundCovers, Polymorph<GroundCover, GroundCoverRectangular>("GroundCoverRectangular"));
        // add other type of canopy here
        
        shpFile = "";
        parsePrimitive<std::string>(false, shpFile, "SHPFile");
        
        shpTreeLayerName = "trees";  // defaults
        parsePrimitive<std::string>(false, shpTreeLayerName, "SHPTreeLayer");
        
        //
        // Process ESRIShapeFile here, but leave extraction of poly
        // building for later in WINDSGeneralData
        //
        SHPData = nullptr;
        if (shpFile != "") {
            // Read polygon node coordinates and building height from shapefile
            SHPData = new ESRIShapefile( shpFile, shpTreeLayerName,  shpPolygons, shpTreeHeight, 1.0 );
            //std::cout << shpPolygons.size() << " " << shpTreeHeight.size() << std::endl;
        } 
        

        

    }
};
