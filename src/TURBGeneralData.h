#pragma once

#include <math.h>
#include <vector>

#include "NetCDFInput.h"
#include "URBGeneralData.h"
#include "TURBWall.h"
#include "TURBWallBuilding.h"
#include "TURBWallTerrain.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class TURBGeneralData 
{
public:
    
    // Defoult
    TURBGeneralData()
    {}
    TURBGeneralData(URBGeneralData*);
    
    // load data at given time instance
    void loadNetCDFData(int);

    // General QUIC Domain Data
    int nx, ny, nz;		/**< number of cells */
    
    //grid information
    std::vector<float> x_fc;
    std::vector<float> x_cc;
    std::vector<float> y_fc;
    std::vector<float> y_cc;
    std::vector<float> z_fc;
    std::vector<float> z_cc;
    
    //mixing length
    std::vector<float> Lm;
    
    // stress stensor
    std::vector<float> tau11;
    std::vector<float> tau12;
    std::vector<float> tau13;
    std::vector<float> tau22;
    std::vector<float> tau23;
    std::vector<float> tau33;
    
    // derived turbulence quantities
    std::vector<float> tke;
    std::vector<float> CoEps; 
    
private:
    
    // input: store here for multiple time instance.
    NetCDFInput* input;
    
};
