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
#include "TypeDefs.hpp"

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
        
        // wind stress and variance information
        std::vector<matrix6> tau, sig;
        std::vector<matrix9> lam;
        
        // CoEps
        std::vector<double>CoEps;
        std::vector<double>tke;
};

#endif