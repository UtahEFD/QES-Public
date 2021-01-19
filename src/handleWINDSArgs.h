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
 * This class handles different commandline options and arguments
 * and places the values into variables. This inherits from Argument Parsing
 */

#include <iostream>
#include "util/ArgumentParsing.h"

enum solverTypes : int
{CPU_Type = 1, DYNAMIC_P = 2, Global_M = 3, Shared_M = 4};

class WINDSArgs : public ArgumentParsing
{
public:

    WINDSArgs();

    ~WINDSArgs() {}

    /*
     * Takes in the commandline arguments and places
     * them into variables.
     *
     * @param argc -number of commandline options/arguments
     * @param argv -array of strings for arguments
     */
    void processArguments(int argc, char *argv[]);


    bool verbose;

    // input files (from the command line)
    std::string quicFile = "";

    // Base name for all NetCDF output files
    std::string netCDFFileBasename = "";

    // flag to turn on/off different modules
    bool solveWind,compTurb;
    int solveType, compareType;

    bool visuOutput,wkspOutput,turbOutput,terrainOut;
    // netCDFFile for standard cell-center vizalization file
    std::string netCDFFileVisu = "";
    // netCDFFile for working field used by Plume
    std::string netCDFFileWksp = "";
    // netCDFFile for turbulence field used by Plume
    std::string netCDFFileTurb = "";
    // filename for terrain output
    std::string filenameTerrain = "";

private:

};
