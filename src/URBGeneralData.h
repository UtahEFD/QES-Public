#pragma once

#include <string>
#include <vector>
#include <netcdf>
#include "NetCDFInput.h"
#include "Args.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

class URBGeneralData
{
public:
    
    // Default
    URBGeneralData() {}
    
    // initializer
    URBGeneralData(Args*);
    
    // load data at given time instance
    void loadNetCDFData(int);
    
    //nt - number of time instance in data
    int nt;
    // time vector
    std::vector<float> t;
    
    /*
      Information below match URBGeneral data class of URB
    */
    
    // General QUIC Domain Data
    int nx, ny, nz;         /**< number of cells */
    float dx, dy, dz;	  /**< Grid resolution*/
    float dxy;		  /**< Minimum value between dx and dy */
    
    // grid information
    std::vector<float> dz_array;
    std::vector<float> x_cc,y_cc,z_cc;
    
    // The following are mostly used for output
    std::vector<int> icellflag;  /**< Cell index flag (0 = Building, 1 = Fluid, 2 = Terrain, 
                                    3 = Upwind_cavity, 4 = Cavity, 5 = Farwake, 6 = Street canyon, 
                                    7 = Cut-cells, 9 = Canopy vegetation, 10 = Sidewall) */
    
    // Terrain data
    std::vector<float> terrain;
    
    /// Declaration of final velocity field components (u,v,w)
    std::vector<float> u,v,w;
    
    /// Declaration of coefficients for SOR solver
    std::vector<float> e,f,g,h,m,n;
    
private:
    
    // input: store here for multiple time instance.
    NetCDFInput* input;
    
    
};
