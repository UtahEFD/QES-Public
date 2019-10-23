//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"
#include <fstream>
#define EPSILON 0.00001   

Dispersion::Dispersion(Urb* urb, Turb* turb, PlumeInputData* PID) {
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

    
    int id=int(srcZ/dz)*ny*nx+int(srcY/dy)*nx+int(srcX/dx);     // appears to be the cell ID of the source point location, used to get the right variance values for the source values
    int idt=int(srcY/dy)*nx+int(srcX/dx);       // doesn't appear to be used
    for(int i=0;i<numPar;i++){
        pos.at(i).x=srcX;   // set the source positions for each particle
        pos.at(i).y=srcY;
        pos.at(i).z=srcZ;
    
        double rann=random::norRan();   // almost didn't see it, but it does use different random numbers for each direction
        prime.at(i).x=turb->sig.at(id).e11 * rann;  // set the values for the source positions for each particle. Might need to add sqrt of the variance to match Brian's code
        rann=random::norRan();
        prime.at(i).y=turb->sig.at(id).e22 * rann;
        rann=random::norRan();
        prime.at(i).z=turb->sig.at(id).e33 * rann;

        // set the initial values for the old stuff
        prime_old.at(i).x = prime.at(i).x;
        prime_old.at(i).y = prime.at(i).y;
        prime_old.at(i).z = prime.at(i).z;

        // set tau_old to the interpolated values for each position
        tau_old.at(i).e11 = turb->tau.at(id).e11;
        tau_old.at(i).e12 = turb->tau.at(id).e12;
        tau_old.at(i).e13 = turb->tau.at(id).e13;
        tau_old.at(i).e22 = turb->tau.at(id).e22;
        tau_old.at(i).e23 = turb->tau.at(id).e23;
        tau_old.at(i).e33 = turb->tau.at(id).e33;

        // set delta_prime to zero for now
        delta_prime.at(i).x = 0.0;
        delta_prime.at(i).y = 0.0;
        delta_prime.at(i).z = 0.0;

        // set isRogue and isActive to true for each particle
        isRogue.at(id) = true;
        isActive.at(id) = true;
        
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