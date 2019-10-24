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
        
        Dispersion(Urb*,Turb*,PlumeInputData*,Eulerian*); // starts by making copies of nx,ny, dx,dy,dz, numParticles, timestep,runTime, and source->posX,posY,posZ
                                                // then calculates the number of timesteps numTimeStep from runTime and timestep.
                                                // then calculates the list of times timeStepStamp
                                                // then sets the pos and prime values for each particle, which are the starting positions and the starting values for each particle
                                                // then calculate the number of particles to release at each time, so the total number to release is split up for each time
                                                // yeah we need to change how this is done cause I want to release once at the start. Need to set time start for release, and time end for release
                                                // then finally set the release times for each and every particle, since they are released constantly over time

        // looks like this is just defining a list of information for Plume to use to know where and when to release different particles.
        // so defining where and when every single particle is released over the entire simulation.
        // looks like there needs to be a lot of work here. I thought the source had a radius as well as a point input, but here we only use a single point location
        // a lot of work needs done to get the source stuff up to par. Right now can only release at a point, and it is ugly to figure out how to use this
        // current method will NOT work for Brian's test cases, as they release all over the entire domain, only at the very start of the simulation
        
        // the way this information is used in Plume right now is as follows:
        // 

        struct matrix {
            double x;
            double y;
            double z;
        };      // why this and not a pos matrix? wait, when do you define it? Is it just sitting here? ah, type is matrix.
  
        double eps;                     // is this machine tolerance, or tke/epps stuff? Doesn't appear to be used. Instead, a compile time variable EPSILON seems to be used
        int numTimeStep;            // this is the number of timesteps of the simulation
        std::vector<double> timeStepStamp;  // this is the list of times for the simulation
        std::vector<vec3> pos, prime;     // pos is the list of particle source positions, prime is the particle source values. Actually, the prime is the velFluct value for a given iteration!
        std::vector<vec3> prime_old;     // this is the velocity fluctuations from the last iteration. They start out the same as the current values initially
        std::vector<matrix6> tau_old;       // this is the stress tensor from the last iteration. Starts out as the values at the initial position, the values for the initial iteration
        std::vector<vec3> delta_prime;    // this is the difference between the current and last iteration of the velocity fluctuations
        std::vector<bool> isRogue;         // this is false until it becomes true. Should not go true. It is whether a particle has gone rogue or not
        std::vector<bool> isActive;         // this is true until it becomes false. If a particle leaves the domain or runs out of mass, this becomes false. Later we will add a method to start more particles when this has become false
        double isRogueCount;        // just a total number of rogue particles per time iteration
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no
        int parPerTimestep;             // this is the number of particles to release at each timestep, used to calculate the release time for each particle
        std::vector<double> tStrt;        // this is the time of release for each set of particles
        
    private:
        
        double dur;         // this is a copy of the input runTime, or the total amount of time to run the simulation for
        int dt;             // this is a copy of the input timestep
        double srcX,srcY,srcZ;  // these are the source point x, y, and z values as copied from the input source information
        int nx,ny;              // these are copies of the Urb grid nx and ny values. Not sure why nz isn't included
        double dx,dy,dz;        // these are copies of the Urb grid dx,dy,dz values.
        int numPar;        // this is a copy of the input numParticles to be released over the whole simulation

};
#endif
