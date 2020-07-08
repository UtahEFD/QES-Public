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
    std::string quicFile = "", demFile = "", iCellOut = "";
    // netCDFFile_vz for standard cell-center vizalization file
    std::string netCDFFileVz = "";
    // netCDFFile_wk for working field use by TURB and Plume
    std::string netCDFFileWk = "";
    bool cellFace, terrainOut, solveWind;
    int solveType, compareType;

    // Calculate Mixing Length - currently takes a while so it is
    // disabled unless this is set.
    bool calcMixingLength;

    int mlSamples;
    
private:
};
