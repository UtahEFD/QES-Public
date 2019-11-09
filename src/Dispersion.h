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
        
        
        int numTimeStep;            // this is the number of timesteps of the simulation
        std::vector<double> timeStepStamp;  // this is the list of times for the simulation
        
        double isRogueCount;        // just a total number of rogue particles per time iteration
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no

        
        // at some point in time, this would probably be better as a vector, so the number can vary for each timestep
        // then, this variable would act like a counter to the full list of particles. Hm, this gets more complex than that when recycling particles
        int parPerTimestep;             // this is the number of particles to release at each timestep, used to calculate the release time for each particle in Dispersion, and also to update the particle loop counter in Plume.
        

        // the sources can set these values, then the other values are set using urb and turb info using these values
        std::vector<vec3> pos;          // pos is the list of particle source positions
        std::vector<double> tStrt;        // this is the time of release for each set of particles
        
        // once positions are known, can set these values
        std::vector<vec3> prime;     // prime is the velFluct value for a given iteration. Starts out as the initial value until a particle is "released" into the domain
        std::vector<vec3> prime_old;     // this is the velocity fluctuations from the last iteration. They start out the same as the current values initially
        std::vector<matrix6> tau_old;       // this is the stress tensor from the last iteration. Starts out as the values at the initial position, the values for the initial iteration
        std::vector<vec3> delta_prime;    // this is the difference between the current and last iteration of the velocity fluctuations
        std::vector<bool> isRogue;         // this is false until it becomes true. Should not go true. It is whether a particle has gone rogue or not
        std::vector<bool> isActive;         // this is true until it becomes false. If a particle leaves the domain or runs out of mass, this becomes false. Later we will add a method to start more particles when this has become false
        

        // going to have a ton of source functions, this first one will be the source factory operator which calls all the other add source functions
        // depending on the type of source. It should be noted that the end goal will be to get rid of the initialization factory, probably by
        // putting all the add source functions into the sources themselves, probably as a function that just finds the initial particle positions and release times
        // probably all the other variables can be set depending on the source positions, maybe, we shall have to see
        void addSources(Sources* sources);       // this function is for adding a generic source of type . . .
        void addSource(SourcePoint* source);   // this is the Continuous SourcePoint type

        void setParticleVals(Turb* turb, Eulerian* eul);
        

    private:
        
        // just realized, what if urb and turb have different grids? For now assume they are the same grid
        int nx,ny;              // these are copies of the Urb grid nx and ny values. Not sure why nz isn't included.    I don't think these are even used
        double dx,dy,dz;        // these are copies of the Urb grid dx,dy,dz values.            I don't think these are even used
        int dt;             // this is a copy of the input timestep
        double simDur;         // this is a copy of the input runTime, or the total amount of time to run the simulation for
        

        int numPar;        // this is a copy of the input numParticles to be released over the whole simulation
        // at some point in time, this needs to become the cumulative number of particles from each source, determined from each source
        // instead of given as a single value that the sources all depend on
        

        // function for finding the largest sig value, which could be used for other similar datatypes if needed
        double maxval(const std::vector<diagonal>& vec);

};
#endif
