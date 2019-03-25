//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#ifndef TURB_HPP
#define TURB_HPP

#include <string>
#include <vector>
#include <netcdf>
#include "Input.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

class Turb {
    
    private:
        
        // netCDF variables
        std::vector<size_t> start;
        std::vector<size_t> count;
        
        NcDim dim_x, dim_y, dim_z, dim_t;
        NcVar var_x, var_y, var_z, var_t;
        NcVar var_u, var_v, var_w, var_icell;
    
    public:
    
        // initializer
        Turb(Input*);
        
        // grid information
        struct Grid {
            int nx, ny, nz, nt;
            std::vector<double>x;
            std::vector<double>y;
            std::vector<double>z;
            std::vector<double>t;
        };
        
        Grid grid;
        
        // wind information
        struct Tau {
            std::vector<double>tau_11;
            std::vector<double>tau_22;
            std::vector<double>tau_33;
            std::vector<double>tau_12;
            std::vector<double>tau_13;
            std::vector<double>tau_23;   
        };
        
        Tau tau;
};

#endif