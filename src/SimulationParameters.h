#pragma once

/*
 * This function contains variables that define information
 * necessary for running the simulation.
 */

#include <string>
#include "util/ParseInterface.h"
#include "Vector3.h"
#include "DTEHeightField.h"

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

        if (demFile != "") {
            std::cout << "Extracting Digital Elevation Data from " << demFile << std::endl;
            DTE_heightField = new DTEHeightField(demFile,
                                                 (*(grid))[0],
                                                 (*(grid))[1] );
            assert(DTE_heightField);
            
            std::cout << "Forming triangle mesh...\n";
            DTE_heightField->setDomain(domain, grid);
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
        // Process ESRIShapeFile here:
        //
        SHPData = nullptr;
        if (shpFile != "") {
            auto buildingsetup = std::chrono::high_resolution_clock::now(); // Start recording execution time

            std::vector <PolyBuilding> poly_buildings;
            std::vector< std::vector <polyVert> > shpPolygons;
            std::vector< std::vector <polyVert> > poly;
            std::vector <float> base_height;            // Base height of buildings
            std::vector <float> effective_height;            // Effective height of buildings
            float corner_height, min_height;
            std::vector <float> building_height;        // Height of buildings

            // Read polygon node coordinates and building height from shapefile
            SHPData = new ESRIShapefile( UID->simParams->shpFile,
                                         UID->simParams->shpBuildingLayerName,
                                         shpPolygons, building_height );



            std::vector<float> shpDomainSize(2), minExtent(2);
            shpFile->getLocalDomain( shpDomainSize );
            shpFile->getMinExtent( minExtent );
            
            float domainOffset[2] = { 0, 0 };
            for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
            {
                // convert the global polys to local domain coordinates
                for (auto lIdx=0; lIdx<shpPolygons[pIdx].size(); lIdx++)
                {
                    shpPolygons[pIdx][lIdx].x_poly = shpPolygons[pIdx][lIdx].x_poly - minExtent[0] ;
                    shpPolygons[pIdx][lIdx].y_poly = shpPolygons[pIdx][lIdx].y_poly - minExtent[1] ;
                }
            }

            std::cout << "num_poly buildings" << shpPolygons.size() << std::endl;
            // Setting base height for buildings if there is a DEM file
            if (DTE_heightField)
            {
                for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
                {
                    // Get base height of every corner of building from terrain height
                    min_height = mesh->getHeight(shpPolygons[pIdx][0].x_poly, shpPolygons[pIdx][0].y_poly);
                    if (min_height<0)
                    {
                        min_height = 0.0;
                    }
                    for (auto lIdx=1; lIdx<shpPolygons[pIdx].size(); lIdx++)
                    {
                        corner_height = mesh->getHeight(shpPolygons[pIdx][lIdx].x_poly, shpPolygons[pIdx][lIdx].y_poly);
                        if (corner_height<min_height && corner_height>0.0)
                        {
                            min_height = corner_height;
                        }
                    }
                    base_height.push_back(min_height);
                }
            }
            else
            {
                for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
                {
                    base_height.push_back(0.0);
                }
            }
            
            for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
            {
                effective_height.push_back (base_height[pIdx]+building_height[pIdx]);
                for (auto lIdx=0; lIdx<shpPolygons[pIdx].size(); lIdx++)
                {
                    shpPolygons[pIdx][lIdx].x_poly += UID->simParams->halo_x;
                    shpPolygons[pIdx][lIdx].y_poly += UID->simParams->halo_y;
                }
            }

            std::cout << "Creating buildings from shapefile...\n";
            // Loop to create each of the polygon buildings read in from the shapefile
            for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
            {
                // Create polygon buildings
                // Pete needs to move this into URBInputData processing BUT
                // instead of adding to the poly_buildings vector, it really
                // needs to be pushed back onto buildings within the
                // UID->buildings structures...
                poly_buildings.push_back (PolyBuilding (shpPolygons[pIdx], building_height[pIdx], base_height[pIdx], nx, ny,
                                                        nz, dx, dy, dz, u0, v0, z));
            }
            
            /// all cell flags should be specific to the TYPE ofbuilding
            // class: canopy, rectbuilding, polybuilding, etc...
            // should setcellflags be part of the .. should be part of URBGeneralD
            for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
            {
                // Call setCellsFlag in the PolyBuilding class to identify building cells
                poly_buildings[pIdx].setCellsFlag ( dx, dy, dz, z, nx, ny, nz, icellflag, UID->simParams->meshTypeFlag, shpPolygons[pIdx], base_height[pIdx], building_height[pIdx]);
            }
        }
    }
    

};
