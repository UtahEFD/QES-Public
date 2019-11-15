//
//  particle.hpp
//  
//  This class represents values stored for each point
//  At a minimum, this needs to have the point position and release time
//  Might want to add in more of the desired variables later
//  Vectors of Particle will need to be vectors of pointers to Particle, making it a bit complex to run
//  but trying to do this as a struct of structs was not working
//
//  Created by Loren Atwood on 11/09/19.
//

#pragma once

#include <vector>
#include "TypeDefs.hpp"

class particle
{

    public:

        // initializer
        particle();

        // destructor
        ~particle();
        

        // the point info variables
        // hm, I'm used to making stuff like this private and creating a bunch of accessor functions
        // so that they all stay the same dimension. But so long as we use them correctly, this isn't a problem
        
        // the sources can set these values, then the other values are set using urb and turb info using these values
        vec3 pos;          // pos is the list of particle source positions
        double tStrt;        // this is the time of release for each set of particles

        // once positions are known, can set these values
        vec3 prime;     // prime is the velFluct value for a given iteration. Starts out as the initial value until a particle is "released" into the domain
        vec3 prime_old;     // this is the velocity fluctuations from the last iteration. They start out the same as the current values initially
        matrix6 tau_old;       // this is the stress tensor from the last iteration. Starts out as the values at the initial position, the values for the initial iteration
        vec3 delta_prime;    // this is the difference between the current and last iteration of the velocity fluctuations
        bool isRogue;         // this is false until it becomes true. Should not go true. It is whether a particle has gone rogue or not
        bool isActive;         // this is true until it becomes false. If a particle leaves the domain or runs out of mass, this becomes false. Later we will add a method to start more particles when this has become false
        

    private:

        

};

