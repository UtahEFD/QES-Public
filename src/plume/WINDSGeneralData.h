#pragma once

#include <string>
#include <vector>
#include <netcdf>
#include "NetCDFInput.h"
#include "Args.hpp"
#include "Wall.h"

#define _USE_MATH_DEFINES
#define MIN_S(x,y) ((x) < (y) ? (x) : (y))
#define MAX_S(x,y) ((x) > (y) ? (x) : (y))

using namespace netCDF;
using namespace netCDF::exceptions;

class WINDSGeneralData
{
public:
    
    // Default
    WINDSGeneralData() {}
    
    // initializer
    WINDSGeneralData(Args*);
    
    // load data at given time instance
    void loadNetCDFData(int);
    
    //nt - number of time instance in data
    int nt;
    // time vector
    std::vector<float> t;
    
    /*
      Information below match WINDSGeneraldata class of QES-Winds
    */
    
    // General QES Domain Data
    int nx, ny, nz;         /**< number of cells */
    float dx, dy, dz;	  /**< Grid resolution*/
    float dxy;		  /**< Minimum value between dx and dy */

    long numcell_cent;       /**< Total number of cell-centered values in domain */
    long numcell_face;       /**< Total number of face-centered values in domain */
    
    // grid information
    std::vector<float> dz_array;
    std::vector<float> x,y,z;
    std::vector<float> z_face;
    
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
 
    // In getWallIndices and wallLogBC
    std::vector<int> wall_right_indices;     /**< Indices of the cells with wall to right boundary condition */
    std::vector<int> wall_left_indices;      /**< Indices of the cells with wall to left boundary condition */
    std::vector<int> wall_above_indices;     /**< Indices of the cells with wall above boundary condition */
    std::vector<int> wall_below_indices;     /**< Indices of the cells with wall bellow boundary condition */
    std::vector<int> wall_back_indices;      /**< Indices of the cells with wall in back boundary condition */
    std::vector<int> wall_front_indices;     /**< Indices of the cells with wall in front boundary condition */
    Wall *wall;

private:
    
    // input: store here for multiple time instance.
    NetCDFInput* input;
    
    
};
