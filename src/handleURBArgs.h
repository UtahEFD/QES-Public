#pragma once

#include <iostream>
#include "ArgumentParsing.h"

enum solverTypes : int
{CPU_Type = 1};

class URBArgs : public ArgumentParsing 
{
public:

    URBArgs();

    ~URBArgs() {}

    void processArguments(int argc, char *argv[]);
    

    bool verbose;
    std::string quicFile = "", netCDFFile = "";
    bool cellFace;
    int solveType;
private:
};