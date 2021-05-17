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

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class WINDSInputData;
class WINDSGeneralData;

class LocalMixingDefault : public LocalMixing
{
private:

protected:

public:

    LocalMixingDefault()
    {}
    ~LocalMixingDefault()
    {}

    /*
     * [FM] This method define the mixing length as the height above the ground
     */
    void defineMixingLength(const WINDSInputData*,WINDSGeneralData*);   

};
