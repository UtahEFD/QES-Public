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
#include "WRFInput.h"
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
    float halo_x = 0.0;
    float halo_y = 0.0;
    float heightFactor = 1.0;

    // DTE - digital elevation model details
    std::string demFile;    // DEM file name
    DTEHeightField* DTE_heightField = nullptr;
    Mesh* DTE_mesh;

    // SHP File parameters
    std::string shpFile;   // SHP file name
    std::string shpBuildingLayerName;
    ESRIShapefile *SHPData = nullptr;
    std::vector< std::vector <polyVert> > shpPolygons;
    std::vector <float> shpBuildingHeight;        // Height of
                                                  // buildings

    // WRF File Parameters
    // - if a WRF element is supplied, we do not load a DEM (at the
    // moment), but extract the DEM from the WRF Fire Mesh.  Metparmas
    // related to stations/sensors are pulled from the wind profile
    // supplied by WRF.
    std::string wrfFile;
    WRFInput *wrfInputData = nullptr;

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
        parseElement< Vector3<int> >(false, domain, "domain");   // when
                                                                 // parseElement
                                                                 // isn't
                                                                 // called,
                                                                 // how
                                                                 // does
                                                                 // this
                                                                 // get allocated?
        parseElement< Vector3<float> >(false, grid, "cellSize");
        parsePrimitive<int>(true, verticalStretching, "verticalStretching");
        parseMultiPrimitives<float>(false, dz_value, "dz_value");
        parsePrimitive<int>(false, totalTimeIncrements, "totalTimeIncrements");
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
        parsePrimitive<float>(false, heightFactor, "heightFactor");

        demFile = "";
        parsePrimitive<std::string>(false, demFile, "DEM");

        shpFile = "";
        parsePrimitive<std::string>(false, shpFile, "SHP");

        wrfFile = "";
        parsePrimitive<std::string>(false, wrfFile, "WRF");

        shpBuildingLayerName = "buildings";  // defaults
        parsePrimitive<std::string>(false, shpBuildingLayerName, "SHPBuildingLayer");

        // Read in height field
        if (wrfFile != "") {
            std::cout << "Processing WRF data for terrain and met param sensors from " << wrfFile << std::endl;
            wrfInputData = new WRFInput( wrfFile );
            std::cout << "WRF Input Data processing completed." << std::endl;

            // In the current setup, grid may NOT be set... be careful
            // may need to initialize it here if nullptr is true for grid

            // utilize the wrf information to construct a
            // DTE_heightfield
            std::cout << "Constructing DTE from WRF Input" << std::endl;
            DTE_heightField = new DTEHeightField(wrfInputData->fmHeight,
                                                 wrfInputData->fm_nx,
                                                 wrfInputData->fm_ny,
                                                 (*(grid))[0],
                                                 (*(grid))[1]);

            // domain = new Vector3<int>( wrfInputData->fm_nx, wrfInputData->fm_nx, 100 );
            DTE_heightField->setDomain(domain, grid);
            DTE_mesh = new Mesh(DTE_heightField->getTris());
            std::cout << "Mesh complete\n";
        }

        // For now wrf and dem are exclusive
        else if (demFile != "") {
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
            // No DEM, so make sure these are null
            DTE_heightField = nullptr;
            DTE_mesh = nullptr;
        }



        //
        // Process ESRIShapeFile here, but leave extraction of poly
        // building for later in URBGeneralData
        //
        SHPData = nullptr;
        if (shpFile != "") {

            // Read polygon node coordinates and building height from shapefile
            SHPData = new ESRIShapefile( shpFile, shpBuildingLayerName,
                                         shpPolygons, shpBuildingHeight, heightFactor );
        }
    }
};
