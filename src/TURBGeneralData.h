#pragma once

#include <math.h>
#include <vector>

#include "Args.hpp"
#include "NetCDFInput.h"
#include "URBGeneralData.h"

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
    TURBGeneralData(Args* arguments, URBGeneralData*);
    
    // load data at given time instance
    void loadNetCDFData(int);

    //nt - number of time instance in data
    int nt;
    // time vector
    std::vector<float> t;
    
    /*
      Information below match TURBGeneral data class of QES-winds
    */

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
    std::vector<float> txx;
    std::vector<float> txy;
    std::vector<float> txz;
    std::vector<float> tyy;
    std::vector<float> tyz;
    std::vector<float> tzz;
    
    // derived turbulence quantities
    std::vector<float> tke;
    std::vector<float> CoEps; 
    
private:
    
    // input: store here for multiple time instance.
    NetCDFInput* input;
    
};
