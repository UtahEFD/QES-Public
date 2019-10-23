//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"
#include <fstream>
#define EPSILON 0.00001   

Dispersion::Dispersion(Urb* urb, Turb* turb, PlumeInputData* PID, Eulerian* eul) {
    std::cout<<"[Dispersion] \t Setting up sources "<<std::endl;
    
    // make local copies
    nx     = urb->grid.nx;
    ny     = urb->grid.ny;
    dx     = urb->grid.dx;
    dy     = urb->grid.dy;
    dz     = urb->grid.dz;
    dt     = PID->simParams->timeStep;
    dur    = PID->simParams->runTime;
    numPar = PID->sources->numParticles;
    srcX   = ((SourcePoint*)PID->sources->sources.at(0))->posX;
    srcY   = ((SourcePoint*)PID->sources->sources.at(0))->posY;
    srcZ   = ((SourcePoint*)PID->sources->sources.at(0))->posZ;
    
    // set up time details
    numTimeStep = std::ceil(dur/dt);
    timeStepStamp.resize(numTimeStep);
    for(int i=0;i<numTimeStep;++i){ 
        timeStepStamp.at(i)=i*dt+dt;
    }
    
    // set up source information
    pos.resize(numPar);
    prime.resize(numPar);

    // set up the initial conditions for the old values too
    prime_old.resize(numPar);
    tau_old.resize(numPar);
    delta_prime.resize(numPar);
    isRogue.resize(numPar);
    isActive.resize(numPar);

    
    
    for(int i=0;i<numPar;i++){
        pos.at(i).x=srcX;   // set the source positions for each particle
        pos.at(i).y=srcY;
        pos.at(i).z=srcZ;

        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(pos.at(i));
    
        double rann=random::norRan();   // almost didn't see it, but it does use different random numbers for each direction

        // get the sigma values from the Eulerian grid for the particle value
        diagonal current_sig = eul->interp3D(turb->sig);

        // now set the initial velocity fluctuations for the particle
        prime.at(i).x = current_sig.e11 * rann;  // set the values for the source positions for each particle. Might need to add sqrt of the variance to match Brian's code
        rann=random::norRan();
        prime.at(i).y = current_sig.e22 * rann;
        rann=random::norRan();
        prime.at(i).z = current_sig.e33 * rann;

        // set the initial values for the old stuff
        prime_old.at(i).x = prime.at(i).x;
        prime_old.at(i).y = prime.at(i).y;
        prime_old.at(i).z = prime.at(i).z;

        // get the tau values from the Eulerian grid for the particle value
        matrix6 current_tau = eul->interp3D(turb->tau);

        // set tau_old to the interpolated values for each position
        tau_old.at(i).e11 = current_tau.e11;
        tau_old.at(i).e12 = current_tau.e12;
        tau_old.at(i).e13 = current_tau.e13;
        tau_old.at(i).e22 = current_tau.e22;
        tau_old.at(i).e23 = current_tau.e23;
        tau_old.at(i).e33 = current_tau.e33;

        // set delta_prime to zero for now
        delta_prime.at(i).x = 0.0;
        delta_prime.at(i).y = 0.0;
        delta_prime.at(i).z = 0.0;

        // set isRogue and isActive to true for each particle
        isRogue.at(i) = true;
        isActive.at(i) = true;
        
    }
    tStrt.resize(numPar);
    
    parPerTimestep = numPar*dt/dur; 
    std::cout<<"[Dispersion] \t Emitting "<<parPerTimestep<< " particles per Time Step"<<std::endl;
    
    int parRel = 0;
    double startTime = dt;
    
    for(int i=0;i<numPar;i++){
      /*when number of particles to be released in a particluar time step reaches total 
        number of particles to be released in that time step, then increase the start time 
        by timestep and set parRel=0*/
        if(parRel==parPerTimestep) {
            startTime=startTime+dt;
            parRel=0;
        }
        tStrt.at(i)=startTime;
        ++parRel;
    }
    
    /*
      Checking if the starting time for the last particle is equal to the duration of
      the simulation (for continous release ONLY)
    */
    if (fabs(tStrt.back()-dur)>EPSILON) {
        std::cerr<<" Error, in start time of the particles"<<std::endl;
        exit(1);
    }
    
}