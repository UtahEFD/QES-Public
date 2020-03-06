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

// may need to forward reference
class TURBGeneralData;

class TURBLocalMixing
{ 
private:
  
    std::vector<int> wall_right_indices, wall_left_indices;
    std::vector<int> wall_back_indices,  wall_front_indices;
    std::vector<int> wall_below_indices, wall_above_indices;

    /*
     * [FM] this is obsolete: methode use ray-tracing & optiX
     * This function compute the distance the brute force way
     * for each fuild cell -> next solid cell serching in 3D
     */
    float getMinDist_brute(unsigned int,URBGeneralData*,TURBGeneralData*);
  
    /*
     * [FM] this is a serial ONLY method
     * This function propagate the distance in fuild cell form
     * the wall for the each solid element 
     * -> this method is relatively inefficient and sould be use 
     * only with small domain
     */
    void getMinDistWall(URBGeneralData*,TURBGeneralData*,int);
  
protected:

public:
  
    TURBLocalMixing()
    {}
    ~TURBLocalMixing()
    {}
  
    /*
     * [FM] This method define the mixing length with the serial 
     * methode (CANNOT be parallelized)
     */
    void defineLength(URBGeneralData*,TURBGeneralData*);   
  
};

