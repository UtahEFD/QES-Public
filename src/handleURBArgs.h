#pragma once

/*
 * This class handles different commandline options and arguments
 * and places the values into variables. This inherits from Argument Parsing
 */

#include <iostream>
#include "util/ArgumentParsing.h"

enum solverTypes : int
{CPU_Type = 1, DYNAMIC_P = 2};

class URBArgs : public ArgumentParsing
{
public:

    URBArgs();

    ~URBArgs() {}

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
    std::string quicFile = "", demFile = "";
    
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
    
    
    // Calculate Mixing Length - currently takes a while so it is
    // disabled unless this is set.
    bool calcMixingLength;
    
private:
};
