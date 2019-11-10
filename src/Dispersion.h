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

#include "pointInfo.hpp"


#include <helper_math.h>

class Dispersion {
    
    public:
        
        Dispersion(Urb*,Turb*,PlumeInputData*,Eulerian*); // starts by making copies of nx,ny, dx,dy,dz, timestep, and runTime
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
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no

        
        // at some point in time, this would probably be better as a vector, so the number can vary for each timestep
        // then, this variable would act like a counter to the full list of particles. Hm, this gets more complex than that when recycling particles
        int parPerTimestep;             // this is the number of particles to release at each timestep, used to calculate the release time for each particle in Dispersion, and also to update the particle loop counter in Plume.
        

        // the sources can set these values, then the other values are set using urb and turb info using these values
        std::vector<pointInfo> pointList;


        // goes through and for each source, runs a virtual function to calculate a pointList with position and release times
        // for each source, adding these pointList values to the overall pointList storage in dispersion
        void addSources(Sources* sources);

        void setParticleVals(Turb* turb, Eulerian* eul);
        

    private:
        
        // just realized, what if urb and turb have different grids? For now assume they are the same grid
        int nx,ny;              // these are copies of the Urb grid nx and ny values. Not sure why nz isn't included.    I don't think these are even used
        double dx,dy,dz;        // these are copies of the Urb grid dx,dy,dz values.            I don't think these are even used
        int dt;             // this is a copy of the input timestep
        double simDur;         // this is a copy of the input runTime, or the total amount of time to run the simulation for
        

        // function for finding the largest sig value, which could be used for other similar datatypes if needed
        double maxval(const std::vector<diagonal>& vec);

};
#endif
