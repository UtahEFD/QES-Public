#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include "LocalMixing.h"
#include "NetCDFInput.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class WINDSInputData;
class WINDSGeneralData;

class LocalMixingNetCDF : public LocalMixing
{
private:

protected:

public:

    LocalMixingNetCDF()
    {}
    ~LocalMixingNetCDF()
    {}

    /*
     * [FM] This method define the mixing length as the height above the ground
     */
    void defineMixingLength(const WINDSInputData*,WINDSGeneralData*);   

};
