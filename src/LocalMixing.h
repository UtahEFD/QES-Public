#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>
#include <netcdf>

#include "NetCDFOutput.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020

*/

class URBInputData;
class URBGeneralData;

class LocalMixing
{
private:
    NetCDFOutput* mixLengthOut;

protected:
    void saveMixingLength(const URBInputData*,URBGeneralData*);

public:
    
    LocalMixing()
    {}
    ~LocalMixing()
    {}
    
    virtual void defineMixingLength(const URBInputData*,URBGeneralData*) = 0;   
};

