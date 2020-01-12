//
//  Urb.hpp
//  
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//  Modified by Loren Atwood on 01/09/20.
//

#ifndef URB_HPP
#define URB_HPP

#include <string>
#include <vector>
#include <netcdf>
#include "Input.hpp"


using namespace netCDF;
using namespace netCDF::exceptions;

class Urb {
    
    private:
        
        // netCDF variables
        // do these need moved into the function itself since they are used only once?
        std::vector<size_t> start;      // used for getVariableData() when it is a grid of values. The starting values in each dimension of the grid of input data. I would guess it is changed each time data is read from the input data
        std::vector<size_t> count,count2d;  // the number of values in each grid dimension of the input data, I guess the default is a 4D structure, but there is a variable for a 2D structure here as well
        
        
    public:
    
        // initializer
        Urb(Input*);    // looks like this just grabs the input urb stuff, and stuffs it into the appropriate variables
                        // the variables may be different than the input ones, so eventually we may want to change that, but is almost there
                        // most of the temporary variables are used as intermediates between the two different types of data structures
        
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
        std::vector<double> icell;   // not sure what this is, probably has to do with the cutcell stuff
        std::vector<double> terrain;   // this has something to do with the input terrain

        // additional grid information
        double urbXstart;    // the urb starting x value. Is not necessarily the urb domain starting x value because it could be cell centered.
        double urbXend;      // the urb ending x value. Is not necessarily the urb domain ending x value because it could be cell centered.
        double urbYstart;    // the urb starting y value. Is not necessarily the urb domain starting y value because it could be cell centered.
        double urbYend;      // the urb ending y value. Is not necessarily the urb domain ending y value because it could be cell centered.
        double urbZstart;    // the urb starting z value. Is not necessarily the urb domain starting z value because it could be cell centered.
        double urbZend;      // the urb ending z value. Is not necessarily the urb domain ending z value because it could be cell centered.
        
        
        // wind information
        std::vector<double> u;     // this is the set of mean wind in the x direction. Probably in m/s
        std::vector<double> v;     // this is the set of mean wind in the y direction. Probably in m/s
        std::vector<double> w;     // this is the set of mean wind in the z direction. Probably in m/s
};

#endif