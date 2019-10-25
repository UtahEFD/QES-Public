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
        pos.at(i).e11=srcX;   // set the source positions for each particle
        pos.at(i).e21=srcY;
        pos.at(i).e31=srcZ;

        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(pos.at(i));
    
        double rann=random::norRan();   // almost didn't see it, but it does use different random numbers for each direction

        // get the sigma values from the Eulerian grid for the particle value
        diagonal current_sig = eul->interp3D(turb->sig,"sigma2");

        // now set the initial velocity fluctuations for the particle
        prime.at(i).e11 = sqrt(current_sig.e11) * rann;  // set the values for the source positions for each particle. Might need to add sqrt of the variance to match Brian's code
        rann=random::norRan();
        prime.at(i).e21 = sqrt(current_sig.e22) * rann;
        rann=random::norRan();
        prime.at(i).e31 = sqrt(current_sig.e33) * rann;

        // set the initial values for the old stuff
        prime_old.at(i).e11 = prime.at(i).e11;
        prime_old.at(i).e21 = prime.at(i).e21;
        prime_old.at(i).e31 = prime.at(i).e31;

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
        delta_prime.at(i).e11 = 0.0;
        delta_prime.at(i).e21 = 0.0;
        delta_prime.at(i).e31 = 0.0;

        // set isRogue to false and isActive to true for each particle
        isRogue.at(i) = false;
        isActive.at(i) = true;
        
    }

    // set the isRogueCount to zero
    isRogueCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*sqrt(maxval(turb->sig));  // might need to write a maxval function, since it has to get the largest value from the entire sig array

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


double Dispersion::maxval(const std::vector<diagonal>& vec)
{
    // set the initial maximum value to a very small number. The idea is to go through each value of the data,
    // setting the current value to the max value each time the current value is bigger than the old maximum value
    double maximumVal = -10e-10;

    double nVals = vec.size();

    for(int idx = 0; idx < nVals; idx++)
    {
        if(vec.at(idx).e11 > maximumVal)
        {
            maximumVal = vec.at(idx).e11;
        }
        if(vec.at(idx).e22 > maximumVal)
        {
            maximumVal = vec.at(idx).e22;
        }
        if(vec.at(idx).e33 > maximumVal)
        {
            maximumVal = vec.at(idx).e33;
        }
    }

    return maximumVal;
    
}