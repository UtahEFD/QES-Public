//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"
#include <fstream>
#define EPSILON 0.00001   

Dispersion::Dispersion(Urb* urb, Turb* turb, PlumeInputData* PID, Eulerian* eul)
{

    std::cout<<"[Dispersion] \t Setting up sources "<<std::endl;
    
    // make local copies
    nx     = urb->grid.nx;
    ny     = urb->grid.ny;
    dx     = urb->grid.dx;
    dy     = urb->grid.dy;
    dz     = urb->grid.dz;
    dt     = PID->simParams->timeStep;
    simDur    = PID->simParams->runTime;

    // have to set the value to something before filling it
    numPar = 0;

    // set up time details
    numTimeStep = std::ceil(simDur/dt);
    timeStepStamp.resize(numTimeStep);
    for(int i = 0; i < numTimeStep; ++i)
    {
        timeStepStamp.at(i) = i*dt + dt;
    }


    // this function goes through each source and adds to the particle list the initial positions and release times
    // this function is an initialization factory calling an add source for each type of source
    // eventually this all needs moved to be a single virtual function inside of sourceKind that is overloaded by each source
    // then the if statements in the initialization factory function will go away
    // I think the new virtual function would need to be called something else though, like generatePointInfo()
    addSources(PID->sources);

    // now that all the sources have added to the particle list, it's time to setup the initial values for each particle using their initial positions
    setParticleVals(turb, eul);


    // set the isRogueCount to zero
    isRogueCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*sqrt(maxval(turb->sig));  // might need to write a maxval function, since it has to get the largest value from the entire sig array

    
    /*
      Checking if the starting time for the last particle is equal to the duration of
      the simulation (for continous release ONLY)
      would need to modify this to check all particles, probably better to just do the check at the end of each source, 
      so it is known which source causes the problem
    */
    if ( fabs(pointList.back().tStrt - simDur) > EPSILON )
    {
        std::cerr << " Error, in start time of the particles" << std::endl;
        exit(1);
    }
    
}


void Dispersion::addSources(Sources* sources)
{
    // first get the number of sources from the sources variable
    int numSources = sources->numSources;

    
    for(int i = 0; i < numSources; i++)
    {
        // all this should work even if there is no points list output, from what I can tell
        
        // get the position and time info for the current source
        // because of polymorphic inheritance, this virtual function should work regardless of the sourceKind
        // the resulting list of points will vary for each source kind
        std::vector<pointInfo> currentPointsList = sources->sources.at(i)->outputPointInfo(dt,simDur);

        // set the number of particles for the current source
        // keeping all these numParticle stuff seperate makes it easier to update the overall point information
        // with the current point information from the current source
        int currentNumPar = currentPointsList.size();

        // update the overall list number of particles
        // keep track of the old number of values so it is easy to update the list in the right way
        int oldNumPar = numPar;
        numPar = numPar + currentNumPar;

        // now resize the overall list number of particles
        // hopefully this works to not get rid of the old values set in the old size locations!
        pointList.resize(numPar);

        // now take the point list information output from the source, and pack it into the overall point list
        for(int i = 0; i < currentNumPar; i++)
        {

            // update the position info
            pointList.at(oldNumPar+i).pos.e11 = currentPointsList.at(i).pos.e11;
            pointList.at(oldNumPar+i).pos.e21 = currentPointsList.at(i).pos.e21;
            pointList.at(oldNumPar+i).pos.e31 = currentPointsList.at(i).pos.e31;

            // now update the time info
            pointList.at(oldNumPar+i).tStrt = currentPointsList.at(i).tStrt;

        }

    }

}


void Dispersion::setParticleVals(Turb* turb, Eulerian* eul)
{

    
    // at this time, should be a list of each and every particle that exists
    // might need to vary this to allow for adding sources after the initial constructor
    for(int i = 0; i < numPar; i++)
    {

        // the size of the vector of pointInfo, the pointList, has already been set. Now just need to fill all the remaining values
        
        
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(pointList.at(i).pos);
    
        double rann = random::norRan();   // almost didn't see it, but it does use different random numbers for each direction

        // get the sigma values from the Eulerian grid for the particle value
        diagonal current_sig = eul->interp3D(turb->sig,"sigma2");

        // now set the initial velocity fluctuations for the particle
        pointList.at(i).prime.e11 = sqrt(current_sig.e11) * rann;  // set the values for the source positions for each particle. Might need to add sqrt of the variance to match Brian's code
        rann=random::norRan();
        pointList.at(i).prime.e21 = sqrt(current_sig.e22) * rann;
        rann=random::norRan();
        pointList.at(i).prime.e31 = sqrt(current_sig.e33) * rann;

        // set the initial values for the old stuff
        pointList.at(i).prime_old.e11 = pointList.at(i).prime.e11;
        pointList.at(i).prime_old.e21 = pointList.at(i).prime.e21;
        pointList.at(i).prime_old.e31 = pointList.at(i).prime.e31;

        // get the tau values from the Eulerian grid for the particle value
        matrix6 current_tau = eul->interp3D(turb->tau);

        // set tau_old to the interpolated values for each position
        pointList.at(i).tau_old.e11 = current_tau.e11;
        pointList.at(i).tau_old.e12 = current_tau.e12;
        pointList.at(i).tau_old.e13 = current_tau.e13;
        pointList.at(i).tau_old.e22 = current_tau.e22;
        pointList.at(i).tau_old.e23 = current_tau.e23;
        pointList.at(i).tau_old.e33 = current_tau.e33;

        // set delta_prime to zero for now
        pointList.at(i).delta_prime.e11 = 0.0;
        pointList.at(i).delta_prime.e21 = 0.0;
        pointList.at(i).delta_prime.e31 = 0.0;

        // set isRogue to false and isActive to true for each particle
        pointList.at(i).isRogue = false;
        pointList.at(i).isActive = true;
        
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