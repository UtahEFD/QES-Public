//
//  Particle.hpp
//  
//  This class represents information stored for each particle
//  Where particles represent Lagrangian data
//
//  Created by Loren Atwood on 11/09/19.
//

#pragma once

#include "Vector3.h"

class Particle
{

public:

    // initializer
    Particle()
    {
    }

    // destructor
    ~Particle()
    {
    }
        

    // the point info variables
    // LA: I'm used to making stuff like this private and creating a bunch of accessor functions
    //  so that they all get edited together. But so long as we use them correctly, this isn't a problem
        
    // values set by emitParticles() by each source
    // the initial position for the particle, to not be changed after the simulation starts
    double xPos_init;          // the initial x component of position for the particle
    double yPos_init;          // the initial y component of position for the particle
    double zPos_init;          // the initial z component of position for the particle

    double tStrt;           // the time of release for the particle
    int sourceIdx;       // the index of the source the particle came from

    // once initial positions are known, can set these values using urb and turb info
    // Initially, the initial x component of position for the particle. 
    // After the solver starts to run, the current x component of position for the particle.
    double xPos;          // x component of position for the particle. 
    double yPos;          // y component of position for the particle. 
    double zPos;          // z component of position for the particle. 

    // The velocity fluctuation for a particle for a given iteration. 
    // Starts out as the initial value until a particle is "released" into the domain
    double uFluct;      // u component 
    double vFluct;      // v component 
    double wFluct;      // w component 

    // The velocity fluctuation for a particle from the last iteration
    double uFluct_old;      // u component
    double vFluct_old;      // v component
    double wFluct_old;      // w component

    // stress tensor from the last iteration (6 component because stress tensor is symetric)
    double txx_old;         // this is the stress in the x direction on the x face from the last iteration
    double txy_old;         // this is the stress in the y direction on the x face from the last iteration 
    double txz_old;         // this is the stress in the z direction on the x face from the last iteration 
    double tyy_old;         // this is the stress in the y direction on the y face from the last iteration
    double tyz_old;         // this is the stress in the z direction on the y face from the last iteration
    double tzz_old;         // this is the stress in the z direction on the z face from the last iteration

    double delta_uFluct;    // this is the difference between the current and last iteration of the uFluct variable
    double delta_vFluct;    // this is the difference between the current and last iteration of the vFluct variable
    double delta_wFluct;    // this is the difference between the current and last iteration of the wFluct variable

    bool isRogue;          // this is false until it becomes true. Should not go true. It is whether a particle has gone rogue or not
    bool isActive;         // this is true until it becomes false.  If a particle leaves the domain or runs out of mass, this becomes false. Later we will add a method to start more particles when this has become false
        

private:

        

};

