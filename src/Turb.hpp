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
        std::vector<size_t> start;      // used for getVariableData() when it is a grid of values. The starting values in each dimension of the grid of input data. I would guess it is changed each time data is read from the input data
        std::vector<size_t> count;      // the number of values in each grid dimension of the input data, I guess the default is a 4D structure
        
        // it doesn't look like any of these are used in Turb.cpp, but maybe they are here in case something ever inherits from this Turb
        NcDim dim_x, dim_y, dim_z, dim_t;   /// not sure what NcDim type is, but it appears to be defined by <netcdf>
        NcVar var_x, var_y, var_z, var_t;   /// not sure what NcVar type is, but it appears to be defined by <netcdf>
        NcVar var_u, var_v, var_w, var_icell;   /// hm, here is the stuff I saw in PlumeInputData.FileOptions. Why is it a repeat of Urb?
    
    public:
    
        // initializer
        Turb(Input*);   // looks like this just grabs the input turb stuff, and stuffs it into this Grid and value data structures
                        // the Grid and value data structures are different than the input ones, so eventually we may want to change that
                        // most of the temporary variables are used as intermediates between the two different types of data structures
        
        // grid information
        struct Grid {               // this is the Turb grid of information
            int nx, ny, nz, nt;     // this is the number of points in each dimension, including time
            double dx, dy, dz;      // this is the difference between points in each dimension, not including time
            std::vector<double>x;   // this is the x values of the grid, probably in m
            std::vector<double>y;   // this is the y values of the grid, probably in m
            std::vector<double>z;   // this is the z values of the grid, probably in m
            std::vector<double>t;   // I'm guessing that x,y, and z are grouped as 1 to nx*ny*nz for each time, probably in s
        };
        
        Grid grid;

        // these values are calculated from the turb data during construction
        // useful to compare these with urb values when debugging
        double domainXstart;    // the urb domain starting x value
        double domainXend;      // the urb domain ending x value
        double domainYstart;    // the urb domain starting y value
        double domainYend;      // the urb domain ending y value
        double domainZstart;    // the urb domain starting z value
        double domainZend;      // the urb domain ending z value

        
        // wind stress and variance information
        std::vector<diagonal> sig;      // this datatype is a diagonal of the stress tensor, (txx, tyy, tzz)
                                        // so this is a vector of three values (e11, e22, e33)
                                        // interestingly, in Brian's code, this is a single value not three separate values
                                        // units are probably m^2/s^2

        std::vector<matrix6> tau;       // this is the only non symmetric parts of the stress tensor (txx, txy, txz, tyy, tyz, tzz)
                                        // so this is a vector of 6 values (e11, e12, e13, e22, e23, e33)
                                        // units are probably m^2/s^2

        std::vector<matrix9> lam;       // this dataype is the full tensor, but the inverted for of it (not symmetric anymore)
                                        // so it is (itxx, itxy, itxz, ityx, ityy, ityz, itzx, itzy, itzz)
                                        // so this is a vector of 6 values (e11, e12, e13, e21, e22, e23, e31, e32, e33)
                                        // I think in Brian's code, this is still symmetric, so we usually only use the 6 components!
                                        // so probably will need to adapt/change this structure at some point in time, or just make
                                        // lam a matrix6 instead of a matrix9 datatype
                                        // units are probably m^2/s^2
        
        // CoEps
        std::vector<double>CoEps;       // this is a single value at the same locations as the sig, tau, and lam. It represents Co*Eps
                                        // where Co is a universal constant and Eps is the mean dissipation rate of turbulent kinetic energy
                                        // because Co is a constant, and the two always go together, Co*Eps is treated as a single value
                                        // units are m^2/s^3? Need to double check this one and all the other units as well

        std::vector<double>tke;         // this is the turbulent kinetic energy, not to be mixed up with Eps. The tke is estimated using tau,
                                        // then Eps is estimated from the tke, Co, and ustar (the friction velocity). CoEps is the Eps derived using
                                        // these things, but then multiplied by another Co. Not sure if this is mean tke or subfilter tke,
                                        // or just the straight up tke. It isn't really used in Brian's code, though it potentially could be used in makeRealizable
                                        // units are ?
};

#endif