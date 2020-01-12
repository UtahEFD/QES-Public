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
#include "PlumeInputData.hpp"

#include <chrono>

using namespace netCDF;
using namespace netCDF::exceptions;

class Plume {
    
    public:
        
        Plume(Urb*,Dispersion*,PlumeInputData*,Output*);    // first makes a copy of the urb grid number of values and the domain size as determined by dispersion
                                                            // then sets the initial data by first calculating the concentration sampling box information for output
                                                            // next copies important input time values and calculates needed time information
                                                            // next sets up the boundary condition functions
                                                            // finally, the output information is setup

        void run(Urb*,Turb*,Eulerian*,Dispersion*,PlumeInputData*,Output*); // has a much cleaner solver now, but at some time needs the boundary conditions adapted to vary for more stuff
                                                                            // also needs two CFL conditions, one for each particle time integration (particles have multiple timesteps smaller than the simulation timestep), and one for the eulerian grid go one cell at a time condition
                                                                            // finally, what output should be normally put out, and what output should only be put out when debugging is super important

        void save(Output*);
        
    private:
        
        // just realized, what if urb and turb have different grids? For now assume they are the same grid
        // variables set during constructor. Notice that the later list of output manager stuff is also setup during the constructor
        int nx;       // a copy of the urb grid nx value
        int ny;       // a copy of the urb grid ny value
        int nz;       // a copy of the urb grid nz value
        double dx;       // a copy of the urb grid dx value, eventually could become an array
        double dy;       // a copy of the urb grid dy value, eventually could become an array
        double dz;       // a copy of the urb grid dz value, eventually could become an array


        // these values are calculated from the urb and turb grids by dispersion
        // they are used for applying boundary conditions at the walls of the domain
        double domainXstart;    // the domain starting x value, a copy of the value found by dispersion
        double domainXend;      // the domain ending x value, a copy of the value found by dispersion
        double domainYstart;    // the domain starting y value, a copy of the value found by dispersion
        double domainYend;      // the domain ending y value, a copy of the value found by dispersion
        double domainZstart;    // the domain starting z value, a copy of the value found by dispersion
        double domainZend;      // the domain ending z value, a copy of the value found by dispersion


        // output concentration box data
        double sCBoxTime;           // a copy of the input startTime, which is the starting time for averaging of the output concentration sampling
        double avgTime;            // this is a copy of the input timeAvg
        int nBoxesX,nBoxesY,nBoxesZ;    // these are copies of the input nBoxesX,Y, and Z. These parameters are the number of boxes to use in the concentration sampling for output
        double lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;     // these are copies of the input parameters boxBoundsX1, boxBoundsX2, boxBoundsY1, ... . These are the upper and lower bounds in each direction of the concentration sampling boxes for output
        double boxSizeX,boxSizeY,boxSizeZ;      // these are the box sizes in each direction, taken by dividing the box bounds by the number of boxes to use, where these boxes are for concentration sampling for output
        double volume;      // this is the volume of the boxes to use in the concentration sampling for output. Is nBoxesX*nBoxesY*nBoxesZ
        std::vector<double> xBoxCen,yBoxCen,zBoxCen;    // I believe these are the list of x,y, and z points for the concentration sampling box information
        std::vector<double> cBox,conc;      // these are the concentration box and concentration values for the simulation


        // input time variables
        double dt;          // this is a copy of the input timeStep
        double simDur;         // this is a copy of the input simDur, or the total amount of time to run the simulation for
        
        // these are the calculated time information needed for the simulation
        int numTimeStep;            // this is the number of timesteps of the simulation, the calculated size of timeStepStamp
        std::vector<double> timeStepStamp;  // this is the list of times for the simulation
        

        double invarianceTol;       // this is the tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
        double C_0;                 // used to separate out CoEps into its separate parts when doing debug output
        int updateFrequency_particleLoop;       // used to know how frequently to print out information during the particle loop of the solver
        int updateFrequency_timeLoop;       // used to know how frequently to print out information during the time loop of the solver
        

        // utility functions for the plume solver
        // hm, this is the one place where it may be helpful to bring back in the complex data types
        void calcInvariants(const double& txx,const double& txy,const double& txz,const double& tyy,const double& tyz,const double& tzz,
                            double& invar_xx,double& invar_yy,double& invar_zz);
        void makeRealizable(double& txx,double& txy,double& txz,double& tyy,double& tyz,double& tzz);
        void invert3(double& A_11,double& A_12,double& A_13,double& A_21,double& A_22,double& A_23,double& A_31,double& A_32,double& A_33);
        void matmult(const double& A_11,const double& A_12,const double& A_13,const double& A_21,const double& A_22,const double& A_23,
                     const double& A_31,const double& A_32,const double& A_33,const double& b_11,const double& b_21,const double& b_31,
                     double& x_11, double& x_21, double& x_31);

        // might need to create multiple versions depending on the selection of boundary condition types by the inputs
        void setBCfunctions(std::string xBCtype,std::string yBCtype,std::string zBCtype);    // a function used at constructor time to set the pointer function to the desired BC type
        void (Plume::*enforceWallBCs_x)(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);  // a pointer to the wallBC function for the x direction. Which function it points at is determined by setBCfunctions and the input xBCtype
        void (Plume::*enforceWallBCs_y)(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);  // a pointer to the wallBC function for the y direction. Which function it points at is determined by setBCfunctions and the input yBCtype
        void (Plume::*enforceWallBCs_z)(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);  // a pointer to the wallBC function for the z direction. Which function it points at is determined by setBCfunctions and the input zBCtype
        void enforceWallBCs_exiting(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);
        void enforceWallBCs_periodic(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);
        void enforceWallBCs_reflection(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd);

        // this is called to set the values whenever it is found that a particle is inactive or rogue
        void setFinishedParticleVals(double& xPos,double& yPos,double& zPos, const bool& isActive, const bool& isRogue);

        
        // functions used to average the output concentrations
        void average(const int, const Dispersion*, const Urb*);     // this one is called right at output. Going to keep. Calculates the concentration averages
        
        void writeSimInfoFile(Dispersion* dis, const double& current_time);
        

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
