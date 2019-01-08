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
    std::string quicFile = "", netCDFFile = "", demFile = "", iCellOut = "";
    bool cellFace, terrainOut, solveWind;
    int solveType, compareType;
private:
};
