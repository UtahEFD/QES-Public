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
#include "LocalMixing.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class LocalMixingSerial : public LocalMixing
{ 
private:
  
    std::vector<int> wall_right_indices, wall_left_indices;
    std::vector<int> wall_back_indices,  wall_front_indices;
    std::vector<int> wall_below_indices, wall_above_indices;
    
    //grid information
    std::vector<float> x_fc,x_cc;
    std::vector<float> y_fc,y_cc;
    std::vector<float> z_fc,z_cc;
    
    /*
     * [FM] this is a serial ONLY method
     * This function propagate the distance in fuild cell form
     * the wall for the each solid element 
     * -> this method is relatively inefficient and sould be use 
     * only with small domain
     */
    void getMinDistWall(URBGeneralData*,int);
    
protected:

public:
    
    LocalMixingSerial()
    {}
    ~LocalMixingSerial()
    {}
    
    /*
     * [FM] This method define the mixing length with the serial 
     * methode (CANNOT be parallelized)
     */
    void defineMixingLength(URBGeneralData*);   
    
};

