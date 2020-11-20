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

class WINDSInputData;
class WINDSGeneralData;

class LocalMixing
{
private:
    NetCDFOutput* mixLengthOut;

protected:
    void saveMixingLength(const WINDSInputData*,WINDSGeneralData*);

public:

    LocalMixing()
    {}
    ~LocalMixing()
    {}

    virtual void defineMixingLength(const WINDSInputData*,WINDSGeneralData*) = 0;   
};

