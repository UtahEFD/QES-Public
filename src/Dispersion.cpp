//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"


Dispersion::Dispersion( PlumeInputData* PID,URBGeneralData* UGD,TURBGeneralData* TGD,Eulerian* eul, const bool& debug_val)
    : pointList(0)  // ???
{
    std::cout<<"[Dispersion] \t Setting up sources "<<std::endl;

    // copy debug information
    debug = debug_val;
    

    // calculate the domain start and end values, needed for source position range checking
    determineDomainSize(eul);

    // make copies of important input time variables
    sim_dt = PID->simParams->timeStep;
    simDur = PID->simParams->simDur;


    // set up time details
    // LA note: the method I am now using to calculate times is kind of strange.
    //  the goal was to find a method that would keep the times going from the simulation startTime 
    //  to the simulation endTime. I found that std::ceil(simDur/dt) gives one timestep too few.
    //  But that is a good thing, cause now the times calculation loop will always start from 0
    //  and stop at one less than the end time for each and every case. So then the endTime just needs
    //  appended to the times loop right after the times calculation loop.
    // LA possible future work: if the startTime stops being zero, I think the method here will still stand,
    //  but instead of using dt*i for the times calculation in the loop, you would need startTime + dt*i.
    nSimTimes = std::ceil(simDur/sim_dt)+1;
    simTimes.resize(nSimTimes);
    for(int sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++)   // end one time early
    {
        simTimes.at(sim_tIdx) = sim_dt*sim_tIdx;
        //std::cout << "simTimes[" << sim_tIdx << "] = \"" << simTimes.at(sim_tIdx) << "\"" << std::endl;
    }
    simTimes.at(nSimTimes-1) = simDur;
    //std::cout << "simTimes[" << nSimTimes-1 << "] = \"" << simTimes.at(nSimTimes-1) << "\"" << std::endl;



    // get sources from input data and add them to the allSources vector
    // this also calls the many check and calc functions for all the input sources
    // !!! note that these check and calc functions have to be called here 
    //  because each source requires extra data not found in the individual source data
    // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
    getInputSources(PID);


    // set the isRogueCount and isNotActiveCount to zero
    isRogueCount = 0.0;
    isNotActiveCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*std::sqrt(getMaxVariance(eul->sig_x,eul->sig_y,eul->sig_z));

    // to make sure the output knows the initial positions and particle sourceIDs for each time iteration
    // ahead of release time, the entire list of particles is generated now, and given initial values
    // this also includes creation of a vector of number of particles to release at a given time
    generateParticleList(TGD,eul);
    
}


void Dispersion::determineDomainSize(Eulerian* eul)
{

    // multiple ways to do this for now. Could just use the turb grid,
    //  or could determine which grid has the smallest and largest value,
    //  or could use input information to determine if they are cell centered or
    //  face centered and use the dx type values to determine the real domain size.
    // We had a discussion that because there are ghost cells on the grid, probably can
    //  just pretend the ghost cells are a halo region that can still be used during plume solver
    //  but ignored when determining output. This could allow particles to reenter the domain as well.

    // for now, I'm just going to use the urb grid, as having differing grid sizes requires extra info for the interp functions
    domainXstart = eul->xStart;
    domainXend = eul->xEnd;
    domainYstart = eul->yStart;
    domainYend = eul->yEnd;
    domainZstart = eul->zStart;
    domainZend = eul->zEnd;
}


void Dispersion::getInputSources(PlumeInputData* PID)
{
    int numSources_Input = PID->sources->sources.size();

    if( numSources_Input == 0 )
    {
        std::cerr << "ERROR (Dispersion::getInputSources): there are no sources in the input file!" << std::endl;
        exit(1);
    }

    // start at zero particles to release and increment as the number per source is found out
    totalParsToRelease = 0;

    for(auto sIdx = 0u; sIdx < numSources_Input; sIdx++)
    {
        // first create the pointer to the input source
        SourceKind *sPtr;

        // now point the pointer at the source
        sPtr = PID->sources->sources.at(sIdx);
        
        // now do anything that is needed to the source via the pointer
        sPtr->setSourceIdx(sIdx);
        sPtr->m_rType->calcReleaseInfo(PID->simParams->timeStep, PID->simParams->simDur);
        sPtr->m_rType->checkReleaseInfo(PID->simParams->timeStep, PID->simParams->simDur);
        sPtr->checkPosInfo(domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend);

        // now determine the number of particles to release for the source and update the overall count
        totalParsToRelease = totalParsToRelease + sPtr->m_rType->m_numPar;

        // now add the pointer that points to the source to the list of sources in dispersion
        allSources.push_back( sPtr );
    }
}


double Dispersion::getMaxVariance(const std::vector<double>& sigma_x_vals,const std::vector<double>& sigma_y_vals,const std::vector<double>& sigma_z_vals)
{
    // set the initial maximum value to a very small number. The idea is to go through each value of the data,
    // setting the current value to the max value each time the current value is bigger than the old maximum value
    double maximumVal = -10e-10;

    
    // go through each vector to find the maximum value
    // each one could potentially be different sizes if the grid is not 3D
    for(int idx = 0; idx < sigma_x_vals.size(); idx++)
    {
        if(sigma_x_vals.at(idx) > maximumVal)
        {
            maximumVal = sigma_x_vals.at(idx);
        }
    }

    for(int idx = 0; idx < sigma_y_vals.size(); idx++)
    {
        if(sigma_y_vals.at(idx) > maximumVal)
        {
            maximumVal = sigma_y_vals.at(idx);
        }
    }

    for(int idx = 0; idx < sigma_z_vals.size(); idx++)
    {
        if(sigma_z_vals.at(idx) > maximumVal)
        {
            maximumVal = sigma_z_vals.at(idx);
        }
    }

    return maximumVal;
    
}

void Dispersion::generateParticleList(TURBGeneralData* TGD, Eulerian* eul)
{

    // Add new particles now
    // - walk over all sources and add the emitted particles from
    // each source to the overall particle list
    for(int sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++)
    {

        std::vector<Particle> nextSetOfParticles;
        for(auto sIdx = 0u; sIdx < allSources.size(); sIdx++)
        {
            int numNewParticles = allSources.at(sIdx)->emitParticles( (float)sim_dt, (float)( simTimes.at(sim_tIdx) ), nextSetOfParticles );
        }
        
        setParticleVals( TGD, eul, nextSetOfParticles );
        
        // append all the new particles on to the big particle
        // advection list
        pointList.insert( pointList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end() );

        // now calculate the number of particles to release for this timestep
        nParsToRelease.push_back(nextSetOfParticles.size());

    }   // end for timeIdx loop

}


void Dispersion::setParticleVals(TURBGeneralData* TGD, Eulerian* eul, std::vector<Particle>& newParticles)
{
    // at this time, should be a list of each and every particle that exists at the given time
    // particles and sources can potentially be added to the list elsewhere
    for(int parIdx = 0; parIdx < newParticles.size(); parIdx++)
    {
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(newParticles.at(parIdx).xPos_init,newParticles.at(parIdx).yPos_init,newParticles.at(parIdx).zPos_init);
        
        // set the positions to be used by the simulation to the initial positions
        newParticles.at(parIdx).xPos = newParticles.at(parIdx).xPos_init;
        newParticles.at(parIdx).yPos = newParticles.at(parIdx).yPos_init;
        newParticles.at(parIdx).zPos = newParticles.at(parIdx).zPos_init;

        // almost didn't see it, but it does use different random numbers for each direction
        double rann = random::norRan();

        // get the sigma values from the Eulerian grid for the particle value
        double current_sig_x = eul->interp3D(eul->sig_x);
        if( current_sig_x == 0.0 )
            current_sig_x = 1e-8;
        double current_sig_y = eul->interp3D(eul->sig_y);
        if( current_sig_y == 0.0 )
            current_sig_y = 1e-8;
        double current_sig_z = eul->interp3D(eul->sig_z);
        if( current_sig_z == 0.0 )
            current_sig_z = 1e-8;
        
        // now set the initial velocity fluctuations for the particle
        // The  sqrt of the variance is to match Bailey's code
        newParticles.at(parIdx).uFluct = std::sqrt(current_sig_x) * rann;
        rann=random::norRan();      // should be randn() matlab equivalent, which is a normally distributed random number
        newParticles.at(parIdx).vFluct = std::sqrt(current_sig_y) * rann;
        rann=random::norRan();
        newParticles.at(parIdx).wFluct = std::sqrt(current_sig_z) * rann;

        // set the initial values for the old velFluct values
        newParticles.at(parIdx).uFluct_old = newParticles.at(parIdx).uFluct;
        newParticles.at(parIdx).vFluct_old = newParticles.at(parIdx).vFluct;
        newParticles.at(parIdx).wFluct_old = newParticles.at(parIdx).wFluct;

        // get the tau values from the Eulerian grid for the particle value
        double current_txx = eul->interp3D(TGD->txx);
        double current_txy = eul->interp3D(TGD->txy);
        double current_txz = eul->interp3D(TGD->txz);
        double current_tyy = eul->interp3D(TGD->tyy);
        double current_tyz = eul->interp3D(TGD->tyz);
        double current_tzz = eul->interp3D(TGD->tzz);

        // set tau_old to the interpolated values for each position
        newParticles.at(parIdx).txx_old = current_txx;
        newParticles.at(parIdx).txy_old = current_txy;
        newParticles.at(parIdx).txz_old = current_txz;
        newParticles.at(parIdx).tyy_old = current_tyy;
        newParticles.at(parIdx).tyz_old = current_tyz;
        newParticles.at(parIdx).tzz_old = current_tzz;

        // set delta_velFluct values to zero for now
        newParticles.at(parIdx).delta_uFluct = 0.0;
        newParticles.at(parIdx).delta_vFluct = 0.0;
        newParticles.at(parIdx).delta_wFluct = 0.0;

        // set isRogue to false and isActive to false for each particle
        // LA note: I want isActive to be true, but this function is now called 
        //  for all particles before they are released so it needs to start out as false
        newParticles.at(parIdx).isRogue = false;
        newParticles.at(parIdx).isActive = false;
        
    }

}
