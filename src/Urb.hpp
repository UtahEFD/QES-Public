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
#include "TypeDefs.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

class Urb {
    
    private:
        
        // netCDF variables
        std::vector<size_t> start;
        std::vector<size_t> count;
        
        NcDim dim_x, dim_y, dim_z, dim_t;
        NcVar var_x, var_y, var_z, var_t;
        NcVar var_u, var_v, var_w, var_icell;
    
    public:
    
        // initializer
        Urb(Input*);
        
        // grid information
        struct Grid {
            int nx, ny, nz, nt;
            double dx, dy, dz;
            std::vector<double>x;
            std::vector<double>y;
            std::vector<double>z;
            std::vector<double>t;
            std::vector<double>icell;   
        };
        
        Grid grid;
        
        // wind information
        std::vector<Wind> wind;
};

#endif