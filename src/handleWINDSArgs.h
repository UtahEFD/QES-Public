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
    // NetCDFFile for fire output
    std::string netCDFFileFire = "";
    
    

private:

};
