//
//  Dispersion.h
//  
//  This class handles dispersion information
//

#ifndef DISPERSION_H
#define DISPERSION_H

#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include "PlumeInputData.hpp"
#include "Eulerian.h"
#include "Random.h"

#include "particle.hpp"


#include <helper_math.h>

class Dispersion {
    
    public:
        
        Dispersion(Urb*,Turb*,PlumeInputData*,Eulerian*,const std::string& debugOutputFolder_val); // starts by making copies of nx,ny, dx,dy,dz, timestep, and runTime
                                                // then calculates the number of timesteps called numTimeStep from runTime and timestep.
                                                // then calculates the list of times called timeStepStamp
                                                // next, it calls addSource() to go through each source and set the initial positions and release times for each particle
                                                // then it calls setParticleVals() to set the initial values for each particle using the list of positions and the urb and turb data
                                                // lastly, some values that are not unique to each particle are set such as the number of rogue particles, called isRogueCount
                                                //  and the velocity threshold used to know if particles are rogue
                                                // finally, and this probably needs adjusted to look through each particle instead of just doing this for the last particle, 
                                                //  the release times are checked to make sure they aren't past the simulation time
                                                
        // looks like this is just defining a list of information for Plume to use to know where and when to release different particles.
        // so defining where and when every single particle is released over the entire simulation.
        // this also holds the full list of particle information, so the dispersion class could probably be renamed to Lagrangian since it is the Lagrangian grid of values
        
        
        int numPar;        // this is the total number of particles to be released over the whole simulation
                            // it is the cumulative number of particles from each source, determined from each source
                            // at some time, need to figure out how to estimate this or update this when sources are added, not by the constructor
        

        int numTimeStep;            // this is the number of timesteps of the simulation
        std::vector<double> timeStepStamp;  // this is the list of times for the simulation
        
        double isRogueCount;        // just a total number of rogue particles per time iteration
        double isActiveCount;       // just a total number of active particles per time iteration
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no

        
        // at some point in time, this would probably be better as a vector, so the number can vary for each timestep
        // then, this variable would act like a counter to the full list of particles. Hm, this gets more complex than that when recycling particles
        std::vector<int> parPerTimestep;             // this is the number of particles to release at each timestep, used to calculate the release time for each particle in Dispersion, and also to update the particle loop counter in Plume.
        

        // 
        // This the storage for all particles
        // 
        // the sources can set these values, then the other values are set using urb and turb info using these values
        std::vector<particle> pointList;


        // ALL Sources that will be used 
        std::vector< SourceKind* > allSources;


        void setParticleVals(Turb* turb, Eulerian* eul, std::vector<particle>& newParticles);

        
        void outputVarInfo_text();


        // this is the output folder for debug variable output
        std::string debugOutputFolder;
        

    private:
        
        // just realized, what if urb and turb have different grids? For now assume they are the same grid
        int nx,ny;              // these are copies of the Urb grid nx and ny values. Not sure why nz isn't included.    I don't think these are even used
        double dx,dy,dz;        // these are copies of the Urb grid dx,dy,dz values.            I don't think these are even used
        double domainXstart;    // a copy of the urb domain starting x value
        double domainXend;      // a copy of the urb domain ending x value
        double domainYstart;    // a copy of the urb domain starting y value
        double domainYend;      // a copy of the urb domain ending y value
        double domainZstart;    // a copy of the urb domain starting z value
        double domainZend;      // a copy of the urb domain ending z value
        double dt;             // this is a copy of the input timestep
        double simDur;         // this is a copy of the input runTime, or the total amount of time to run the simulation for
        

        // this function takes the sources from PlumeInputData and puts them into the allSources vector found in dispersion
        // this also calls the check metadata function for the input sources before adding them to the list.
        // the check metadata function should already have been called for all the other sources during the specialized constructor phases used to create them.
        void getInputSources(PlumeInputData* PID);


        // function for finding the largest sig value, which could be used for other similar datatypes if needed
        double maxval(const std::vector<diagonal>& vec);

};
#endif
