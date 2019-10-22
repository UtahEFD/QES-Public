//
//  Urb.hpp
//  
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//

#ifndef URB_HPP
#define URB_HPP

#include <string>
#include <vector>
#include <netcdf>
#include "Input.hpp"
#include "TypeDefs.hpp" // is this even used here? doesn't seem like it
                        // yes it is, there is a type <Wind>, a struct of u, v, w that is separate from the urb grid winds
                        // looks like a copy of the grid winds, but in a different data structure/format
                        // since no other spot in the entire code uses <Wind>, should we just move it into here?

using namespace netCDF;
using namespace netCDF::exceptions;

class Urb {
    
    private:
        
        // netCDF variables
        std::vector<size_t> start;      // used for getVariableData() when it is a grid of values. The starting values in each dimension of the grid of input data. I would guess it is changed each time data is read from the input data
        std::vector<size_t> count,count2d;  // the number of values in each grid dimension of the input data, I guess the default is a 4D structure, but there is a variable for a 2D structure here as well
        
        // it doesn't look like any of these are used in Urb.cpp, but maybe they are here in case something ever inherits from this Urb
        NcDim dim_x, dim_y, dim_z, dim_t;   /// not sure what NcDim type is, but it appears to be defined by <netcdf>
        NcVar var_x, var_y, var_z, var_t;   /// not sure what NcVar type is, but it appears to be defined by <netcdf>
        NcVar var_u, var_v, var_w, var_icell;   /// hm, here is the stuff I saw in PlumeInputData.FileOptions
        
    public:
    
        // initializer
        Urb(Input*);    // looks like this just grabs the input urb stuff, and stuffs it into this Grid and Wind structure
                        // the Grid and Wind data structures are different than the input ones, so eventually we may want to change that
                        // most of the temporary variables are used as intermediates between the two different types of data structures
        
        // grid information
        struct Grid {               // this is the Urb grid of information
            int nx, ny, nz, nt;     // this is the number of points in each dimension, including time
            double dx, dy, dz;      // this is the difference between points in each dimension, not including time
            std::vector<double>x;   // this is the x values of the grid, probably in m
            std::vector<double>y;   // this is the y values of the grid, probably in m
            std::vector<double>z;   // this is the z values of the grid, probably in m
            std::vector<double>t;   // I'm guessing that x,y, and z are grouped as 1 to nx*ny*nz for each time, probably in s
            std::vector<double>icell;   // not sure what this is, probably has to do with the cutcell stuff
            std::vector<double>terrain;   // this has something to do with the input terrain
        };
        
        Grid grid;
        
        // wind information
        std::vector<Wind> wind;     // this is the set of mean wind values. The data structure is defined in TypeDefs.hpp. Wind is just a struct of u,v,w values. Probably in m/s
};

#endif