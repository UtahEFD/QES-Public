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
    if (fabs(tStrt.back()-simDur)>EPSILON) {
        std::cerr<<" Error, in start time of the particles"<<std::endl;
        exit(1);
    }
    
}


void Dispersion::addSources(Sources* sources)
{
    // first get the number of sources from the sources variable
    int numSources = sources->numSources;

    // also grab the number of particles, since that is where this variable is stored for now. I want to eventually move this to the individual sources
    // then numPar will represent a cumulative number from each source
    numPar = sources->numParticles;

    
    for(int i = 0; i < numSources; i++)
    {
        // right now I use if statements because this is an initialization factory, but at some time, I hope to move these function calls
        // into the source classes themselves, as a single virtual function that is overloaded by each source
        //   oh crap, the name of the source type isn't even stored, so this won't even work
        // so I'm not going to do if statements for now, just going to call the function that does the source type
        // even worse, without using an if statement I've created a new pointer each iteration that I technically want to get rid of
        // not sure if this would cause memory leak. For now, I'll hope it gets deleted at each iteration
        // but hopefully this doesn't mean I'm accidentally deleting the original pointer to the source?
        // I think it should at least run still as is
        SourcePoint* currentSource = (SourcePoint*)sources->sources.at(i);

        addSource(currentSource);

    }

}


void Dispersion::addSource(SourcePoint* source)
{
    // for now this is an okay spot, when adding multiple sources, this variable needs updated/added to, or at least
    // needs to be a temporary one for unpacking these values into the full list of values
    // oh crap, I want an individual value for this for each source rather than a single value for all sources
    // but it isn't setup this way in the input files yet, so I can't do that. But right now I'm forcing a source type
    // so the value isn't inside that storage yet either
    // so going to comment this out for now
    //numPar = source->numParticles;
    
    // these are the source point x, y, and z values from the input source information
    double srcX = source->posX;
    double srcY = source->posY;
    double srcZ = source->posZ;
    
    
    // set up source information sizes
    pos.resize(numPar);
    tStrt.resize(numPar);
    

    // now set source positions
    for(int i = 0; i < numPar; i++)
    {

        pos.at(i).e11 = srcX;   // set the source positions for each particle
        pos.at(i).e21 = srcY;
        pos.at(i).e31 = srcZ;

    }

    // setup the number of particles per timestep
    // this should eventually be adjusting this value, not adding to it
    parPerTimestep = numPar*dt/simDur;
    std::cout << "[Dispersion] \t Emitting " << parPerTimestep << " particles per Time Step" << std::endl;
    
    int parRel = 0;
    double startTime = dt;
    
    // now set source release times
    for(int i = 0; i < numPar; i++)
    {
      /*when number of particles to be released in a particluar time step reaches total 
        number of particles to be released in that time step, then increase the start time 
        by timestep and set parRel=0*/
        if( parRel == parPerTimestep )
        {
            startTime = startTime + dt;
            parRel = 0;
        }
        tStrt.at(i) = startTime;
        ++parRel;
    }


}

void Dispersion::setParticleVals(Turb* turb, Eulerian* eul)
{

    // now set source value size
    prime.resize(numPar);

    // set up the initial conditions for the old values too
    prime_old.resize(numPar);
    tau_old.resize(numPar);
    delta_prime.resize(numPar);
    isRogue.resize(numPar);
    isActive.resize(numPar);

    
    // at this time, should be a list of each and every particle that exists
    // might need to vary this to allow for adding sources after the initial constructor
    for(int i = 0; i < numPar; i++)
    {

        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(pos.at(i));
    
        double rann = random::norRan();   // almost didn't see it, but it does use different random numbers for each direction

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