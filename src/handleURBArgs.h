#pragma once

#include <iostream>
#include "util/ArgumentParsing.h"

enum solverTypes : int
{CPU_Type = 1, DYNAMIC_P = 2};

class URBArgs : public ArgumentParsing 
{
public:

    URBArgs();

    ~URBArgs() {}

    void processArguments(int argc, char *argv[]);
    

    bool verbose;
    std::string quicFile = "", netCDFFile = "", demFile = "";
    bool cellFace, iCellOut, terrainOut, solveWind;
    int solveType;
private:
};
