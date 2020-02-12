//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//  Modified by Loren Atwood on 01/09/2020.
//

#ifndef TURB_HPP
#define TURB_HPP

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <netcdf>


#include "Input.hpp"
#include "NetCDFInput.h"


// LA are these needed here???
using namespace netCDF;
using namespace netCDF::exceptions;

class Turb {
    
    private:
        
        // netCDF variables
        // do these need moved into the function itself since they are used only once?
        std::vector<size_t> start;      // used for getVariableData() when it is a grid of values. The starting values in each dimension of the grid of input data. I would guess it is changed each time data is read from the input data
        std::vector<size_t> count;      // the number of values in each grid dimension of the input data, I guess the default is a 4D structure
        
        
    public:
    
        // initializer
        // looks like this just grabs the input turb stuff, and stuffs it into the appropriate variables
        // the variables may be different than the input ones, so eventually we may want to change that, but is almost there
        // most of the temporary variables are used as intermediates between the two different types of data structures
        Turb(Input* input, const bool& debug);

        // initializer
        // looks like this just grabs the input turb stuff, and stuffs it into the appropriate variables
        // the variables may be different than the input ones, so eventually we may want to change that, but is almost there
        // most of the temporary variables are used as intermediates between the two different types of data structures
        Turb(NetCDFInput* input, const bool& debug);
        
        
        // urb grid information
        int nx;     // this is the number of points in the x dimension
        int ny;     // this is the number of points in the y dimension
        int nz;     // this is the number of points in the z dimension
        int nt;     // this is the number of times for which the x,y, and z values are repeated
        double dx;      // this is the difference between points in the x dimension, eventually could become an array
        double dy;      // this is the difference between points in the y dimension, eventually could become an array
        double dz;      // this is the difference between points in the z dimension, eventually could become an array
        std::vector<double> x;   // this is the x values of the grid, probably in m
        std::vector<double> y;   // this is the y values of the grid, probably in m
        std::vector<double> z;   // this is the z values of the grid, probably in m
        std::vector<double> t;   // I'm guessing that x,y, and z are grouped as 1 to nx*ny*nz for each time, probably in s

        
        // additional grid information
        double turbXstart;    // the turb starting x value. Is not necessarily the turb domain starting x value because it could be cell centered.
        double turbXend;      // the turb ending x value. Is not necessarily the turb domain ending x value because it could be cell centered.
        double turbYstart;    // the turb starting y value. Is not necessarily the turb domain starting y value because it could be cell centered.
        double turbYend;      // the turb ending y value. Is not necessarily the turb domain ending y value because it could be cell centered.
        double turbZstart;    // the turb starting z value. Is not necessarily the turb domain starting z value because it could be cell centered.
        double turbZend;      // the turb ending z value. Is not necessarily the turb domain ending z value because it could be cell centered.

        
        // this is the only non symmetric parts of the stress tensor (txx, txy, txz, tyy, tyz, tzz)
        // currently called tau11, tau12, tau13, tau22, tau23, tau33 in CUDA-TURB. I prefer the other name.
        // units are probably m^2/s^2
        std::vector<double> txx;
        std::vector<double> txy;
        std::vector<double> txz;
        std::vector<double> tyy;
        std::vector<double> tyz;
        std::vector<double> tzz;

          
        // this is the variance information. Technically should be the same thing as txx, tyy, and tzz.
        // in Bailey's code this is a single value, not three separate values
        // units are probably m^2/s^2
        std::vector<double> sig_x;
        std::vector<double> sig_y;
        std::vector<double> sig_z;


        // CoEps
        std::vector<double>CoEps;       // this is a single value at the same locations as the sig, tau, and lam. It represents Co*Eps
                                        // where Co is a universal constant and Eps is the mean dissipation rate of turbulent kinetic energy
                                        // because Co is a constant, and the two always go together, Co*Eps is treated as a single value
                                        // units are m^2/s^3? Need to double check this one and all the other units as well

        std::vector<double>tke;         // this is the turbulent kinetic energy, not to be mixed up with Eps. The tke is estimated using tau,
                                        // then Eps is estimated from the tke, Co, and ustar (the friction velocity). CoEps is the Eps derived using
                                        // these things, but then multiplied by another Co. Not sure if this is mean tke or subfilter tke,
                                        // or just the straight up tke. It isn't really used in Bailey's code, though it potentially could be used in makeRealizable
                                        // units are ?
};

#endif