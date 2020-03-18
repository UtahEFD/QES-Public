#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include "URBGeneralData.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class LocalMixing
{ 
private:
  
protected:

public:
  
    LocalMixing()
    {}
    ~LocalMixing()
    {}
  
    virtual void defineMixingLength(URBGeneralData*) = 0;   
  
};

