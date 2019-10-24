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
        
        Plume(Urb*,Dispersion*,PlumeInputData*,Output*);    // appears to set the initial data by first calculating the concentration sampling box information for output
                                                            // this includes making internal copies of simple parameters from the inputs, urb, and dispersion
                                                            // after setting up concentration sampling box info, copies dispersion information
                                                            // finally, the output information is setup
                                                            //
                                                            // this stuff seems fine and dandy, I see the point of making copies of single value variables. That being said, it is strange that some of the vector dispersion information is also copied
                                                            // so then why use dispersion stuff in the run of the solver? Copy it all, or none seems to make sense to me. Unless maybe the single values.
                                                            // why is PlumeInputData still used for the run then? Again, copy it all, or none seems to make sense.
                                                            //
                                                            // I guess at some point in time, it might make sense to reorder a few things in this constructor
                                                            // I already reordered the variables in the private section to match how they are set in the constructor
                                                            // I guess the order does make some sense, except for a few here and there, like tStepInp and avgTime. Need more separation between the variables in the constructor. Maybe first load all needed variables in order of the different classes they are loaded from, then calculate stuff from them just after

        void run(Urb*,Turb*,Eulerian*,Dispersion*,PlumeInputData*,Output*); // interestingly, because the particles are only allowed to move one eulerian grid cell at a time, interpolation is avoided
                                                                            // but I bet that makes applying boundary conditions nasty. That being said, I kind of want to throw it all out, including the dispersion stuff, and start over.
                                                                            // But instead, I'm going to make a second copy of this function, keeping what seems to matter or not, like how new particles are added, or how interpolation is avoided
                                                                            // or how particles only step one grid point at a time. But my end goal is to completely shift a lot of stuff.
        void save(Output*);
        
    private:
        
        // variables set during constructor. Notice that the later list of output manager stuff is also setup during the constructor
        int nx,ny,nz;       // these are copies of the Urb grid nx, ny, and nz values
        int numPar;        // this is a copy of the input numParticles to be released over the whole simulation
        int nBoxesX,nBoxesY,nBoxesZ;    // these are copies of the input nBoxesX,Y, and Z. These parameters are the number of boxes to use in the concentration sampling for output
        double boxSizeX,boxSizeY,boxSizeZ;      // these are the box sizes in each direction, taken by dividing the box bounds by the number of boxes to use, where these boxes are for concentration sampling for output
        double volume;      // this is the volume of the boxes to use in the concentration sampling for output. Is nBoxesX*nBoxesY*nBoxesZ
        double lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;     // these are copies of the input parameters boxBoundsX1, boxBoundsX2, boxBoundsY1, ... . These are the upper and lower bounds in each direction of the concentration sampling boxes for output
        double quanX,quanY,quanZ;           // funny, these appear to be the same thing as the boxSize variables
        std::vector<double> xBoxCen,yBoxCen,zBoxCen;    // I believe these are the list of x,y, and z points for the concentration sampling box information
        double tStepInp,avgTime;            // these are copies of input parameters timeStep and timeAvg. Since timeStep is the integration timestep and timeAvg is the output concentration averaging time, it is strange to me to pull them out next to each other in this way. Also, the resulting names seem terrible.
        double sCBoxTime;           // a copy of the input startTime, which is the starting time for averaging of the output concentration sampling
        int numTimeStep;        // a copy of the dispersion number of timesteps for the simulation
        std::vector<double> tStrt;  // a copy of the dispersion tStrt, which is the time of release for each set of particles to release in the simulation
        std::vector<double> timeStepStamp;  // a copy of the dispersion timeStepStamp, which is the list of times for the simulation
        int parPerTimestep;     // a copy of the dispersion parPerTimestep, which is the number of particles to release per timestep
        std::vector<double> cBox,conc;      // these are the concentration box and concentration values for the simulation


        int tStep;  // this is the integration timestep loop counter. Why the heck it is out here instead of in the loop is beyond me
        
        int loopExt=0;


        // still need to figure out how this is going to work, especially with the data structures
        vec3 calcInvariants(const matrix6& tau);
        matrix6 makeRealizable(const matrix6& tau,const double& invarianceTol);

        // this one might be fun to figure out. I'm used to having multiple outputs and that is not so in C++
        void enforceBCs(double&);

        
        // ironically, almost all of these functions are used only by "reflection", which isn't even called now
        // reflection is normally called right after the particles have been advected and checked, as a way to
        // handle wall boundary conditions
        void average(const int, const Dispersion*, const Urb*);     // this one is called right at output. Going to keep. Calculates the concentration averages
        void outputConc();      // doesn't appear to exist anymore
        void reflection(double&, double&, const double&, const  double&,  const double&, const double&
        		,double&,double&,const Eulerian*,const Urb*,const int&,const int&,const int&,double&,double&);      // this thing is almost bigger and longer than plume! There is a simplified version as if it is just Brian's code, but is not enough
        double dot(const pos&, const pos&); // calculates the dot product of two pos datatypes. Shouldn't this be put in where the pos datatype is defined?
        pos normalize(const pos&); // calculates the norm of a pos datatype. Shouldn't this be put in where the pos datatype is defined?
        pos VecScalarMult(const pos&,const double&);    // calculates a pos datatype multipled by a scalar. Again, shoudn't this go where the pos datatype is defined?
        pos reflect(const pos&,const pos&);     // ah, here is the smaller reflect function. Shouldn't this go where the pos datatype is defined?
        pos posSubs(const pos&,const pos&);     // subtract two pos vector values. Shouldn't this go where the pos datatype is defined?
        pos posAdd(const pos&,const pos&);      // add two pos vector values. Shouldn't this go where the pos datatype is defined?
        double distance(const pos&,const pos&);     // calculate the distance between two pos vectors. Shouldn't this go where the pos datatype is defined?
        
        // these are just some standard utility functions not specific to a datatype
        double min(double[],int);
        double max(double[],int);
        
        // output manager
        // looks like the main variables to output are concentration at each x,y, and z position for each time
        // where the values are only for the concentration sampling boxes and concentration sampling times as calculated during the constructor setup
        int output_counter=0;
        double timeOut=0;
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