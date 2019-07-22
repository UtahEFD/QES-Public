#pragma once

/*
 * This function contains variables that define information
 * necessary for running the simulation.
 */

#include <string>
#include "util/ParseInterface.h"
#include "Vector3.h"
#include "DTEHeightField.h"
#include "ESRIShapefile.h"
#include "Mesh.h"

class SimulationParameters : public ParseInterface
{
private:


public:
    Vector3<int>* domain;
    Vector3<float>* grid;
    int verticalStretching;
    std::vector<float> dz_value;
    int totalTimeIncrements;
    int rooftopFlag;
    int upwindCavityFlag;
    int streetCanyonFlag;
    int streetIntersectionFlag;
    int wakeFlag;
    int sidewallFlag;
    int maxIterations;
    int residualReduction;
    float domainRotation;
    float UTMx;
    float UTMy;
    int UTMZone;
    int UTMZoneLetter;
    int meshTypeFlag;
    float halo_x;
    float halo_y;

    // DTE - digital elevation model details
    std::string demFile;    // DEM file name
    DTEHeightField* DTE_heightField = nullptr;
    Mesh* DTE_mesh;

    // SHP File parameters
    std::string shpFile;   // SHP file name
    std::string shpBuildingLayerName;
    ESRIShapefile *SHPData = nullptr;


    SimulationParameters()
    {
        UTMx = 0.0;
        UTMy = 0.0;
        UTMZone = 0;
        UTMZoneLetter = 0;
    }

    ~SimulationParameters()
    {
        // close the scanner
        if (DTE_heightField)
            DTE_heightField->closeScanner();
    }
    

    virtual void parseValues()
    {
        parseElement< Vector3<int> >(true, domain, "domain");
        parseElement< Vector3<float> >(true, grid, "cellSize");
        parsePrimitive<int>(true, verticalStretching, "verticalStretching");
        parseMultiPrimitives<float>(false, dz_value, "dz_value");
        parsePrimitive<int>(true, totalTimeIncrements, "totalTimeIncrements");
        parsePrimitive<int>(true, rooftopFlag, "rooftopFlag");
        parsePrimitive<int>(true, upwindCavityFlag, "upwindCavityFlag");
        parsePrimitive<int>(true, streetCanyonFlag, "streetCanyonFlag");
        parsePrimitive<int>(true, streetIntersectionFlag, "streetIntersectionFlag");
        parsePrimitive<int>(true, wakeFlag, "wakeFlag");
        parsePrimitive<int>(true, sidewallFlag, "sidewallFlag");
        parsePrimitive<int>(true, maxIterations, "maxIterations");
        parsePrimitive<int>(true, residualReduction, "residualReduction");
        parsePrimitive<int>(true, meshTypeFlag, "meshTypeFlag");
        parsePrimitive<float>(true, domainRotation, "domainRotation");
        parsePrimitive<float>(false, UTMx, "UTMx");
        parsePrimitive<float>(false, UTMy, "UTMy");
        parsePrimitive<int>(false, UTMZone, "UTMZone");
        parsePrimitive<int>(false, UTMZoneLetter, "UTMZoneLetter");
        parsePrimitive<float>(false, halo_x, "halo_x");
        parsePrimitive<float>(false, halo_y, "halo_y");
        demFile = "";
        parsePrimitive<std::string>(false, demFile, "DEM");

        shpFile = "";
  	parsePrimitive<std::string>(false, shpFile, "SHP");

        shpBuildingLayerName = "buildings";  // defaults
        parsePrimitive<std::string>(false, shpBuildingLayerName, "SHPBuildingLayer");

        // Read in height field
        if (demFile != "") {
            std::cout << "Extracting Digital Elevation Data from " << demFile << std::endl;
            DTE_heightField = new DTEHeightField(demFile,
                                                 (*(grid))[0],
                                                 (*(grid))[1] );
            assert(DTE_heightField);
            
            std::cout << "Forming triangle mesh...\n";
            DTE_heightField->setDomain(domain, grid);
            DTE_mesh = new Mesh(DTE_heightField->getTris());
            std::cout << "Mesh complete\n";
        }
        else {
            std::cerr << "Error: No dem file specified in input\n";
        }
        
#if 0
        if (arguments.terrainOut) {
            if (DTEHF) {
                std::cout << "Creating terrain OBJ....\n";
                DTEHF->outputOBJ("terrain.obj");
                std::cout << "OBJ created....\n";
            }
            else {
                std::cerr << "Error: No dem file specified as input\n";
                return -1;
            }
        }
#endif

        
        // 
        // Process ESRIShapeFile here, but leave extraction of poly
        // building for later in URBGeneralData
        //
        SHPData = nullptr;
        if (shpFile != "") {
            std::vector< std::vector <polyVert> > shpPolygons;
            std::vector <float> building_height;        // Height of
                                                        // buildings

            // Read polygon node coordinates and building height from shapefile
            SHPData = new ESRIShapefile( shpFile, shpBuildingLayerName,
                                         shpPolygons, building_height );
        }
    }
};

