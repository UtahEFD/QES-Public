//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"
#include <fstream>
#define EPSILON 0.00001   

Dispersion::Dispersion(Urb* urb, Turb* turb, Eulerian* eul, PlumeInputData* PID) {
    std::cout<<"[Dispersion] \t Setting up sources "<<std::endl;
    
    // make local copies
    nx     = urb->grid.nx;
    ny     = urb->grid.ny;
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
    int id=int(srcZ)*ny*nx+int(srcY)*nx+int(srcX);
    for(int i=0;i<numPar;i++){
        pos.at(i).x=srcX;
        pos.at(i).y=srcY;
        pos.at(i).z=srcZ;
    
        double rann=random::norRan();
        prime.at(i).x=turb->sig.at(id).e11 * rann;
        rann=random::norRan();
        prime.at(i).y=turb->sig.at(id).e22 * rann;
        rann=random::norRan();
        prime.at(i).z=turb->sig.at(id).e33 * rann;
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