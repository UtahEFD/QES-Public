//
//  Plume.hpp
//  
//  This class handles plume model
//

#ifndef PLUME_H
#define PLUME_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include "Output.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Eulerian.h"
#include "Dispersion.h"
#include "TypeDefs.hpp"
#include "PlumeInputData.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

class Plume {
    
    public:
        
        Plume(Urb*,Dispersion*,PlumeInputData*,Output*);    // first makes a copy of the urb grid info
                                                            // then sets the initial data by first calculating the concentration sampling box information for output
                                                            // next copies important input time information for dispersion plus copies of the dispersion info
                                                            // finally, the output information is setup
                                                            
        void run(Urb*,Turb*,Eulerian*,Dispersion*,PlumeInputData*,Output*); // has a much cleaner solver now, but at some time needs the boundary conditions adapted to vary for more stuff
                                                                            // also needs two CFL conditions, one for each particle time integration (particles have multiple timesteps smaller than the simulation timestep), and one for the eulerian grid go one cell at a time condition
                                                                            // finally, what output should be normally put out, and what output should only be put out when debugging is super important
        void save(Output*);
        
    private:
        
        // just realized, what if urb and turb have different grids? For now assume they are the same grid
        // variables set during constructor. Notice that the later list of output manager stuff is also setup during the constructor
        int nx,ny,nz;       // these are copies of the Urb grid nx, ny, and nz values

        // these values are calculated from the urb data during construction
        // they are used for applying boundary conditions at the walls of the domain
        double domainXstart;    // the urb domain starting x value
        double domainXend;      // the urb domain ending x value
        double domainYstart;    // the urb domain starting y value
        double domainYend;      // the urb domain ending y value
        double domainZstart;    // the urb domain starting z value
        double domainZend;      // the urb domain ending z value

        // output concentration box data
        double sCBoxTime;           // a copy of the input startTime, which is the starting time for averaging of the output concentration sampling
        double avgTime;            // this is a copy of the input timeAvg
        int nBoxesX,nBoxesY,nBoxesZ;    // these are copies of the input nBoxesX,Y, and Z. These parameters are the number of boxes to use in the concentration sampling for output
        double boxSizeX,boxSizeY,boxSizeZ;      // these are the box sizes in each direction, taken by dividing the box bounds by the number of boxes to use, where these boxes are for concentration sampling for output
        double volume;      // this is the volume of the boxes to use in the concentration sampling for output. Is nBoxesX*nBoxesY*nBoxesZ
        double lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;     // these are copies of the input parameters boxBoundsX1, boxBoundsX2, boxBoundsY1, ... . These are the upper and lower bounds in each direction of the concentration sampling boxes for output
        double quanX,quanY,quanZ;           // funny, these appear to be the same thing as the boxSize variables
        std::vector<double> xBoxCen,yBoxCen,zBoxCen;    // I believe these are the list of x,y, and z points for the concentration sampling box information
        std::vector<double> cBox,conc;      // these are the concentration box and concentration values for the simulation

        // input data used for dispersion
        double dt;          // this is a copy of the input timeStep
        int numTimeStep;        // a copy of the dispersion number of timesteps for the simulation
        
        // some dispersion variables
//        int numPar;        // this is a copy of the input numParticles to be released over the whole simulation
        std::vector<double> tStrt;  // a copy of the dispersion tStrt, which is the time of release for each set of particles to release in the simulation
        std::vector<double> timeStepStamp;  // a copy of the dispersion timeStepStamp, which is the list of times for the simulation
        std::vector<int> parPerTimestep;     // a copy of the dispersion parPerTimestep, which is the number of particles to release per timestep
        

        // still need to figure out how this is going to work, especially with the data structures
        vec3 calcInvariants(const matrix6& tau);
        matrix6 makeRealizable(const matrix6& tau,const double& invarianceTol);
        matrix9 invert3(const matrix9& A);
        vec3 matmult(const matrix9& Ainv,const vec3& b);

        // might need to create multiple versions depending on the selection of boundary condition types by the inputs
        void enforceWallBCs(double& xPos,double& yPos,double& zPos,bool &isActive);

        
        // functions used to average the output concentrations
        void average(const int, const Dispersion*, const Urb*);     // this one is called right at output. Going to keep. Calculates the concentration averages
        
        

        // output manager
        // looks like the main variables to output are concentration at each x,y, and z position for each time
        // where the values are only for the concentration sampling boxes and concentration sampling times as calculated during the constructor setup
        int output_counter = 0;
        double timeOut = 0;
        std::vector<NcDim> dim_scalar_t;
        std::vector<NcDim> dim_scalar_z;
        std::vector<NcDim> dim_scalar_y;
        std::vector<NcDim> dim_scalar_x;
        std::vector<NcDim> dim_vector;
        
        struct AttScalarDbl {
            double* data;
            std::string name;
            std::string long_name;
            std::string units;
            std::vector<NcDim> dimensions;
        };
        
        struct AttVectorDbl {
            std::vector<double>* data;
            std::string name;
            std::string long_name;
            std::string units;
            std::vector<NcDim> dimensions;
        };
        std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
        std::map<std::string,AttVectorDbl> map_att_vector_dbl; 
        std::vector<AttScalarDbl> output_scalar_dbl;       
        std::vector<AttVectorDbl> output_vector_dbl;

};
#endif
