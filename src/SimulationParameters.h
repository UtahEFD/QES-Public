/*
 * QES-Winds
 *
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 *
 */


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
    int verticalStretching = 0;
    std::vector<float> dz_value;
    int totalTimeIncrements;
    int rooftopFlag = 1;
    int upwindCavityFlag = 2;
    int streetCanyonFlag = 1;
    int streetIntersectionFlag = 0;
    int wakeFlag = 2;
    int sidewallFlag = 1;
    int maxIterations = 500;
    double tolerance = 1e-9;
    float domainRotation = 0;
    float UTMx;
    float UTMy;
    int UTMZone;
    int UTMZoneLetter;
    int meshTypeFlag = 0;
    float halo_x = 0.0;
    float halo_y = 0.0;
    float heightFactor = 1.0;

    int readCoefficientsFlag = 0;
    std::string coeffFile;

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



    enum DomainInputType {
        DEMOnly,
        UNKNOWN
    };
    DomainInputType m_domIType;

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
        parseElement< Vector3<int> >(false, domain, "domain");
        parseElement< Vector3<float> >(false, grid, "cellSize");
        parsePrimitive<int>(false, verticalStretching, "verticalStretching");
        parseMultiPrimitives<float>(false, dz_value, "dz_value");
        parsePrimitive<int>(false, totalTimeIncrements, "totalTimeIncrements");
        parsePrimitive<int>(false, rooftopFlag, "rooftopFlag");
        parsePrimitive<int>(false, upwindCavityFlag, "upwindCavityFlag");
        parsePrimitive<int>(false, streetCanyonFlag, "streetCanyonFlag");
        parsePrimitive<int>(false, streetIntersectionFlag, "streetIntersectionFlag");
        parsePrimitive<int>(false, wakeFlag, "wakeFlag");
        parsePrimitive<int>(false, sidewallFlag, "sidewallFlag");
        parsePrimitive<int>(false, maxIterations, "maxIterations");
        parsePrimitive<double>(false, tolerance, "tolerance");
        parsePrimitive<int>(false, meshTypeFlag, "meshTypeFlag");
        parsePrimitive<float>(false, domainRotation, "domainRotation");
        parsePrimitive<float>(false, UTMx, "UTMx");
        parsePrimitive<float>(false, UTMy, "UTMy");
        parsePrimitive<int>(false, UTMZone, "UTMZone");
        parsePrimitive<int>(false, UTMZoneLetter, "UTMZoneLetter");
        parsePrimitive<float>(false, halo_x, "halo_x");
        parsePrimitive<float>(false, halo_y, "halo_y");
        parsePrimitive<float>(false, heightFactor, "heightFactor");
        parsePrimitive<int>(false, readCoefficientsFlag, "readCoefficientsFlag");

        coeffFile = "";
        parsePrimitive<std::string>(false, coeffFile, "COEFF");

        demFile = "";
        parsePrimitive<std::string>(false, demFile, "DEM");

        shpFile = "";
        parsePrimitive<std::string>(false, shpFile, "SHP");

        shpBuildingLayerName = "buildings";  // defaults
        parsePrimitive<std::string>(false, shpBuildingLayerName, "SHPBuildingLayer");

        // Determine which use case to use for WRF/DEM combinations
        if (demFile != "") {
            // Only DEM Specified - nothing set for WRF Input
            m_domIType = DEMOnly;
        }
        else {
            m_domIType = UNKNOWN;
        }

        //
        // Process the data files based on the state determined above
        //
        
        if (m_domIType == DEMOnly) {
            std::cout << "Extracting Digital Elevation Data from " << demFile << std::endl;
            DTE_heightField = new DTEHeightField(demFile,
                                                 (*(grid))[0],(*(grid))[1],
                                                  UTMx, UTMy, (*(domain))[0],(*(domain))[1]);
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
        // building for later in WINDSGeneralData
        //
        SHPData = nullptr;
        if (shpFile != "") {

            // Read polygon node coordinates and building height from shapefile
            SHPData = new ESRIShapefile( shpFile, shpBuildingLayerName,
                                         shpPolygons, shpBuildingHeight, heightFactor );
        }
    }
};
