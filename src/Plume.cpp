//  Plume.cpp
//
//  
//  This class handles plume model
//

#include "Plume.hpp"

Plume::Plume( PlumeInputData* PID, URBGeneralData* UGD, TURBGeneralData* TGD, Eulerian* eul, Args* arguments) 
    : pointList(0)
{
    
    std::cout<<"[Plume] \t Setting up simulation details "<<std::endl;
    
    // copy debug information
    doLagrDataOutput = arguments->doLagrDataOutput;
    outputSimInfoFile = arguments->doSimInfoFileOutput;
    outputFolder = arguments->outputFolder;
    caseBaseName = arguments->caseBaseName;
    debug = arguments->debug;

    // make local copies of the urb nVals for each dimension
    nx = UGD->nx;
    ny = UGD->ny;
    nz = UGD->nz;
    dx = UGD->dx;
    dy = UGD->dy;
    dz = UGD->dz;

    // get the domain start and end values, needed for wall boundary condition application
    domainXstart = eul->xStart;
    domainXend = eul->xEnd;
    domainYstart = eul->yStart;
    domainYend = eul->yEnd;
    domainZstart = eul->zStart;
    domainZend = eul->zEnd;
    
    // make copies of important dispersion time variables
    sim_dt = PID->simParams->timeStep;
    simDur = PID->simParams->simDur;
    
    nSimTimes = std::ceil(simDur/sim_dt)+1;
    simTimes.resize(nSimTimes);
    for(int sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++)   // end one time early
    {
        simTimes.at(sim_tIdx) = sim_dt*sim_tIdx;
        //std::cout << "simTimes[" << sim_tIdx << "] = \"" << simTimes.at(sim_tIdx) << "\"" << std::endl;
    }
    simTimes.at(nSimTimes-1) = simDur;
    

    // other important time variables not from dispersion
    CourantNum = PID->simParams->CourantNum;

    // make copy of dispersion number of particles to release at each simulation timestep
    // Note it is one less than times because the time loop ends one time early.
    //nParsToRelease.resize(nSimTimes-1);    // first need to get the vector size right for the copy. 

    // set additional values from the input
    invarianceTol = PID->simParams->invarianceTol;
    C_0 = PID->simParams->C_0;
    updateFrequency_timeLoop = PID->simParams->updateFrequency_timeLoop;
    updateFrequency_particleLoop = PID->simParams->updateFrequency_particleLoop;
    
    // set the isRogueCount and isNotActiveCount to zero
    isRogueCount = 0;
    isNotActiveCount = 0;

     // get sources from input data and add them to the allSources vector
    // this also calls the many check and calc functions for all the input sources
    // !!! note that these check and calc functions have to be called here 
    //  because each source requires extra data not found in the individual source data
    // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
    getInputSources(PID);
    
    // to make sure the output knows the initial positions and particle sourceIDs for each time iteration
    // ahead of release time, the entire list of particles is generated now, and given initial values
    // this also includes creation of a vector of number of particles to release at a given time
    //generateParticleList(TGD,eul);
    
    /* setup boundary condition functions */

    // now get the input boundary condition types from the inputs
    std::string xBCtype = PID->BCs->xBCtype;
    std::string yBCtype = PID->BCs->yBCtype;
    std::string zBCtype = PID->BCs->zBCtype;

    // now set the boundary condition function for the plume runs, 
    // and check to make sure the input BCtypes are legitimate
    setBCfunctions(xBCtype,yBCtype,zBCtype);

    // now set the wall reflection function
    if(PID->BCs->wallReflection == "doNothing") {
        wallReflection = &Plume::wallReflectionDoNothing; 
    } else if(PID->BCs->wallReflection == "setInactive") {
        wallReflection = &Plume::wallReflectionSetToInactive; 
    } else if(PID->BCs->wallReflection == "stairstepReflection") {
        wallReflection = &Plume::wallReflectionFullStairStep; 
    } else {
        // this should not happend 
        std::cerr  << "[ERROR] unknown wall reflection setting" << std::endl;
        exit(EXIT_FAILURE);
        
    }
    
}

// LA note: in this whole section, the idea of having single value temporary storage instead of just referencing values
//  directly from the dispersion class seems a bit strange, but it makes the code easier to read cause smaller variable names.
//  Also, it is theoretically faster?
void Plume::run(URBGeneralData* UGD, TURBGeneralData* TGD, Eulerian* eul, std::vector<QESNetCDFOutput*> outputVec)
{
    std::cout << "[Plume] \t Advecting particles " << std::endl;
    
    // get the threshold velocity fluctuation to define rogue particles
    vel_threshold = eul->vel_threshold;
    
    // //////////////////////////////////////////
    // TIME Stepping Loop
    // for every simulation time step
    // //////////////////////////////////////////
    
    if( debug == true ) {
        // start recording the amount of time it takes to perform the simulation time integration loop
        timers.startNewTimer("simulation time integration loop");
        
        // start additional timers that need to be reset at different times during the following loops
        // LA future work: probably should wrap all of these in a debug if statement
        timers.startNewTimer("advection loop");
        timers.startNewTimer("particle iteration");
    }
    
    
    // because particle list is the desired size before the simulation, and the number of particles to move changes
    // each time, need to set the loop counter for the number of particles to move before the simulation time loop
    // !!! note that this is dispersion's value so that the output can get the number of released particles right!
    nParsReleased = 0;
    
    // want to output the particle information for the first timestep for where particles are without moving
    // so going to temporarily set nParsReleased to the number of particles released at the first time
    // do an output for the first time, then put the value back to zero so the particle loop will work correctly
    // this means I need to set the isActive value to true for the first set of particles right here
    /*
      nParsReleased = nParsToRelease.at(0);
      for( int parIdx = 0; parIdx < nParsToRelease.at(0); parIdx++ ) {
      pointList[parIdx].isActive = true;
      }
      for(size_t id_out=0;id_out<outputVec.size();id_out++) {
      outputVec.at(id_out)->save(simTimes.at(0));
      }
      nParsReleased = 0;
    */
    
    // LA note: that this loop goes from 0 to nTimes-2, not nTimes-1. This is because
    //  a given time iteration is calculating where particles for the current time end up for the next time
    //  so in essence each time iteration is calculating stuff for one timestep ahead of the loop.
    //  This also comes out in making sure the numPar to release makes sense. A list of times from 0 to 10 with timestep of 1
    //  means that nTimes is 11. So if the loop went from times 0 to 10 in that case, if 10 pars were released each time, 110 particles, not 100
    //  particles would end up being released.
    // LA note on debug timers: because the loop is doing stuff for the next time, and particles start getting released at time zero,
    //  this means that the updateFrequency needs to match with tStep+1, not tStep. At the same time, the current time output to consol
    //  output and to function calls need to also be set to tStep+1.
    // FMargairaz -> need clean-up
    for(int sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++) {
        
        // need to release new particles
        // Add new particles to the number to move
        // !!! note that the updated number of particles is dispersion's value 
        //  so that the output can get the number of released particles right
        int nPastPars = nParsReleased;
        int nParsToRelease = generateParticleList((float)simTimes.at(sim_tIdx), TGD, eul); 
        //nParsReleased = nPastPars + nParsToRelease;
        nParsReleased = pointList.size();
        //nParsReleased = nPastPars + nParsToRelease.at(sim_tIdx);
        
        // need to set the new particles isActive values to true
        //for( int parIdx = nPastPars; parIdx < nParsReleased; parIdx++ ) {
        //    pointList[parIdx].isActive = true;
        //}
      
        
        // only output the information when the updateFrequency allows and when there are actually released particles
        /*
        if(  nParsToRelease.at(sim_tIdx) != 0 && ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 )  )
        {
            std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) << "\". finished emitting \"" 
                      << nParsToRelease.at(sim_tIdx) << "\" particles from \"" << allSources.size() 
                      << "\" sources. Total numParticles = \"" << nParsReleased << "\"" << std::endl;
        }
        */

        if( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 )
        {
            std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) << "\". finished emitting \"" 
                      << nParsToRelease << "\" particles from \"" << allSources.size() 
                      << "\" sources. Total numParticles = \"" << nParsReleased << "\"" << std::endl;
        }

        
        // Move each particle for every simulation time step
        // Advection Loop
        
        // start recording the amount of time it takes to advect each set of particles for a given simulation timestep,
        //  but only output the result when updateFrequency allows
        // LA future work: would love to put this into a debug if statement wrapper
        if( debug == true ) {
            if( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) {
                timers.resetStoredTimer("advection loop");
            }
        }
        
        // get the isRogue and isNotActive count from the dispersion class for use in each particle iteration
        //int isRogueCount = dis->isRogueCount;
        //int isNotActiveCount = dis->isNotActiveCount;
        
        for( int parIdx = 0; parIdx < nParsReleased; parIdx++ ) {
           
            // first check to see if the particle should even be advected and skip it if it should not be advected
            if( pointList[parIdx].isActive == true ) {
                
                // call to the main particle adection function (in separate file: AdvectParticle.cpp)
                /*
                  this function is advencing the particle
                  -> status is returned is dis->pointList[parIdx].isRogue and dis->pointList[parIdx].isActive 
                 */
                advectParticle(sim_tIdx, parIdx, UGD, TGD, eul);
                
                // now update the isRogueCount and isNotActiveCount
                if(pointList[parIdx].isRogue == true) {
                    isRogueCount = isRogueCount + 1;
                }
                if(pointList[parIdx].isActive == false) {
                    isNotActiveCount = isNotActiveCount + 1;
                }
                                
                
                // get the amount of time it takes to advect a single particle, but only output the result when updateFrequency allows
                if( debug == true ) {
                    if(  ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) 
                         && ( parIdx % updateFrequency_particleLoop == 0 || parIdx == pointList.size()-1 )  ) {
                        std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) 
                                  << "\", par[" << parIdx << "]. finished particle iteration" << std::endl;
                        timers.printStoredTime("particle iteration");
                    }
                }
            }   // if isActive == true and isRogue == false
        } // for(int parIdx = 0; parIdx < dis->nParsReleased; parIdx++ )
        
        // netcdf output for a given simulation timestep
        // note that the first time is already output, so this is the time the loop iteration 
        //  is calculating, not the input time to the loop iteration
        for(size_t id_out=0;id_out<outputVec.size();id_out++) {
            outputVec.at(id_out)->save(simTimes.at(sim_tIdx+1));
        }
        
        // output the time, isRogueCount, and isNotActiveCount information for all simulations,
        //  but only when the updateFrequency allows
        if( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) {
            std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) << "\". finished advection iteration. isRogueCount = \"" 
                      << isRogueCount << "\", isNotActiveCount = \"" << isNotActiveCount << "\"" << std::endl;
            // output advection loop runtime if in debug mode
            if( debug == true ) {
                timers.printStoredTime("advection loop");
            }
        }
        
        if(isNotActiveCount > 0) {
            scrubParticleList();
        }
        
        
        //
        // Pete's notes:
        // For all particles that need to be removed from the particle
        // advection, remove them now
        //
        // Purge the advection list of all the unneccessary particles....
        // 
        // Loren's Notes: for now I want to keep them for the thesis work information and debugging
        //  Also, would it not be easier to do at the start of the loop rather than at the end?
        //  Need to think more on this when we get to it.
        
        
    } // end of loop: for(sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++)
    
    // DEBUG - get the amount of time it takes to perform the simulation time integration loop
    if( debug == true ) {
        std::cout << "finished time integration loop" << std::endl;
        // Print out elapsed execution time
        timers.printStoredTime("simulation time integration loop");
    }
    
    // only outputs if the required booleans from input args are set
    // LA note: the current time put in here is one past when the simulation time loop ends
    //  this is because the loop always calculates info for one time ahead of the loop time.
    writeSimInfoFile(simTimes.at(nSimTimes-1));
    
    return;
}


double Plume::calcCourantTimestep(const double& u,const double& v,const double& w,const double& timeRemainder)
{
    // if the Courant Number is set to 0.0, we want to exit using the timeRemainder (first time through that is the simTime)
    if( CourantNum == 0.0 ) {
        return timeRemainder;
    }
    
    // set the output dt_par val to the timeRemainder
    // then if any of the Courant number values end up smaller, use that value instead
    double dt_par = timeRemainder;
    
    // LA-note: what to do if the velocity fluctuation is zero?
    //  I forced them to zero to check dt_x, dt_y, and dt_z would get values of "inf". 
    //  It ends up keeping dt_par as the timeRemainder
    double dt_x = CourantNum*dx/std::abs(u);
    double dt_y = CourantNum*dy/std::abs(v);
    double dt_z = CourantNum*dz/std::abs(w);
    
    // now find which dt is the smallest one of the Courant Number ones, or the timeRemainder
    // if any dt is smaller than the already chosen output value set that dt to the output dt value
    if( dt_x < dt_par ) {
        dt_par = dt_x;
    }
    if( dt_y < dt_par ) {
        dt_par = dt_y;
    }
    if( dt_z < dt_par ) {
        dt_par = dt_z;
    }
    
    return dt_par;
}

void Plume::getInputSources(PlumeInputData* PID)
{
    int numSources_Input = PID->sources->sources.size();

    if( numSources_Input == 0 ) {
        std::cerr << "ERROR (Dispersion::getInputSources): there are no sources in the input file!" << std::endl;
        exit(1);
    }

    // start at zero particles to release and increment as the number per source is found out
    totalParsToRelease = 0;

    for(auto sIdx = 0u; sIdx < numSources_Input; sIdx++) {
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

void Plume::generateParticleList(TURBGeneralData* TGD, Eulerian* eul)
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

int Plume::generateParticleList(float currentTime,TURBGeneralData* TGD, Eulerian* eul)
{
    
    // Add new particles now
    // - walk over all sources and add the emitted particles from
    
    std::vector<Particle> nextSetOfParticles;
    int numNewParticles = 0;
    for(auto sIdx = 0u; sIdx < allSources.size(); sIdx++) {
        numNewParticles += allSources.at(sIdx)->emitParticles( (float)sim_dt, currentTime, nextSetOfParticles);
    }
    
    setParticleVals( TGD, eul, nextSetOfParticles );
    
    // append all the new particles on to the big particle
    // advection list
    pointList.insert( pointList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end() );
    
    // now calculate the number of particles to release for this timestep

    return numNewParticles;
    
}

void Plume::scrubParticleList()
{
    for(auto it = pointList.begin(); it != pointList.end();) {
        if(it->isActive == false) {
            it = pointList.erase(it);
        } else {
            ++it;
        }
    }
    return;
}

void Plume::setParticleVals(TURBGeneralData* TGD, Eulerian* eul, std::vector<Particle>& newParticles)
{
    // at this time, should be a list of each and every particle that exists at the given time
    // particles and sources can potentially be added to the list elsewhere
    for(int parIdx = 0; parIdx < newParticles.size(); parIdx++)
    {
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindex_cellVar(newParticles.at(parIdx).xPos_init,newParticles.at(parIdx).yPos_init,newParticles.at(parIdx).zPos_init);
        
        // set the positions to be used by the simulation to the initial positions
        newParticles.at(parIdx).xPos = newParticles.at(parIdx).xPos_init;
        newParticles.at(parIdx).yPos = newParticles.at(parIdx).yPos_init;
        newParticles.at(parIdx).zPos = newParticles.at(parIdx).zPos_init;

        // get the sigma values from the Eulerian grid for the particle value
        double current_sig_x = eul->interp3D_cellVar(eul->sig_x);
        if( current_sig_x == 0.0 )
            current_sig_x = 1e-8;
        double current_sig_y = eul->interp3D_cellVar(eul->sig_y);
        if( current_sig_y == 0.0 )
            current_sig_y = 1e-8;
        double current_sig_z = eul->interp3D_cellVar(eul->sig_z);
        if( current_sig_z == 0.0 )
            current_sig_z = 1e-8;
        
        // now set the initial velocity fluctuations for the particle
        // The  sqrt of the variance is to match Bailey's code
        double rann = random::norRan();        // use different random numbers for each direction
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
        double current_txx = eul->interp3D_cellVar(TGD->txx);
        double current_txy = eul->interp3D_cellVar(TGD->txy);
        double current_txz = eul->interp3D_cellVar(TGD->txz);
        double current_tyy = eul->interp3D_cellVar(TGD->tyy);
        double current_tyz = eul->interp3D_cellVar(TGD->tyz);
        double current_tzz = eul->interp3D_cellVar(TGD->tzz);

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
        newParticles.at(parIdx).isActive = true;
        
    }

}


void Plume::calcInvariants(const double& txx,const double& txy,const double& txz,
                           const double& tyy,const double& tyz,const double& tzz,
                           double& invar_xx,double& invar_yy,double& invar_zz)
{
    // since the x doesn't depend on itself, can just set the output without doing any temporary variables
    // (copied from Bailey's code)
    invar_xx = txx + tyy + tzz;
    invar_yy = txx*tyy + txx*tzz + tyy*tzz - txy*txy - txz*txz - tyz*tyz;
    invar_zz = txx*(tyy*tzz - tyz*tyz) - txy*(txy*tzz - tyz*txz) + txz*(txy*tyz - tyy*txz);
}

void Plume::makeRealizable(double& txx,double& txy,double& txz,double& tyy,double& tyz,double& tzz)
{
    // first calculate the invariants and see if they are already realizable
    // the calcInvariants function modifies the values directly, so they always need initialized to something before being sent 
    // into said function to be calculated
    double invar_xx = 0.0;
    double invar_yy = 0.0;
    double invar_zz = 0.0;
    calcInvariants(txx,txy,txz,tyy,tyz,tzz,  invar_xx,invar_yy,invar_zz);
    
    if( invar_xx > invarianceTol && invar_yy > invarianceTol && invar_zz > invarianceTol ) {
        return;     // tau is already realizable
    }
    
    // since tau is not already realizable, need to make it realizeable
    // start by making a guess of ks, the subfilter scale tke
    // I keep wondering if we can use the input Turb->tke for this or if we should leave it as is
    double b = 4.0/3.0*(txx + tyy + tzz);   // also 4.0/3.0*invar_xx 
    double c = txx*tyy + txx*tzz + tyy*tzz - txy*txy - txz*txz - tyz*tyz;   // also invar_yy
    double ks = 1.01*(-b + std::sqrt(b*b - 16.0/3.0*c)) / (8.0/3.0);
    
    // if the initial guess is bad, use the straight up invar_xx value
    if( ks < invarianceTol || isnan(ks) ) {
        ks = 0.5*std::abs(txx + tyy + tzz);  // also 0.5*abs(invar_xx)
    }
    
    // to avoid increasing tau by more than ks increasing by 0.05%, use a separate stress tensor
    // and always increase the separate stress tensor using the original stress tensor, only changing ks for each iteration
    // notice that through all this process, only the diagonals are really increased by a value of 0.05% of the subfilter tke ks
    // start by initializing the separate stress tensor
    double txx_new = txx + 2.0/3.0*ks;
    double txy_new = txy;
    double txz_new = txz;
    double tyy_new = tyy + 2.0/3.0*ks;
    double tyz_new = tyz;
    double tzz_new = tzz + 2.0/3.0*ks;
    
    calcInvariants(txx_new,txy_new,txz_new,tyy_new,tyz_new,tzz_new,  invar_xx,invar_yy,invar_zz);
    
    // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till tau is realizable
    // or if too many iterations go on, give a warning.
    // I've had trouble with this taking too long
    //  if it isn't realizable, so maybe another approach for when the iterations are reached might be smart
    int iter = 0;
    while( (invar_xx < invarianceTol || invar_yy < invarianceTol || invar_zz < invarianceTol) && iter < 1000 ) {
        iter = iter + 1;
        
        // increase subfilter tke by 5%
        ks = ks*1.050;      
        
        // note that the right hand side is not tau_new, to force tau to only increase by increasing ks
        txx_new = txx + 2.0/3.0*ks;
        tyy_new = tyy + 2.0/3.0*ks;
        tzz_new = tzz + 2.0/3.0*ks;
        
        calcInvariants(txx_new,txy_new,txz_new,tyy_new,tyz_new,tzz_new,  invar_xx,invar_yy,invar_zz);
    }
    
    if( iter == 999 ) {
        std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizble.";
    }
    
    // now set the output actual stress tensor using the separate temporary stress tensor
    txx = txx_new;
    txy = txy_new;
    txz = txz_new;
    tyy = tyy_new;
    tyz = tyz_new;
    tzz = tzz_new;
    
}

void Plume::invert3(double& A_11,double& A_12,double& A_13,double& A_21,double& A_22,
                    double& A_23,double& A_31,double& A_32,double& A_33)
{
    // note that with Bailey's code, the input A_21, A_31, and A_32 are zeros even though they are used here
    // at least when using this on tau to calculate the inverse stress tensor. This is not true when calculating the inverse A matrix
    // for the Ax=b calculation
    
    // now calculate the determinant
    double det = A_11*(A_22*A_33 - A_23*A_32) - A_12*(A_21*A_33 - A_23*A_31) + A_13*(A_21*A_32 - A_22*A_31);
    
    // check for near zero value determinants
    // LA future work: I'm still debating whether this warning needs to be limited by the updateFrequency information
    //  if so, how would we go about limiting that info? Would probably need to make the loop counter variables actual data members of the class
    if(std::abs(det) < 1e-10) {
        std::cout << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
        std::cout << "abs(det) = \"" << std::abs(det) << "\",  A_11 = \"" << A_11 << "\", A_12 = \"" << A_12 << "\", A_13 = \"" 
                  << A_13 << "\", A_21 = \"" << A_21 << "\", A_22 = \"" << A_22 << "\", A_23 = \"" << A_23 << "\", A_31 = \"" 
                  << A_31 << "\" A_32 = \"" << A_32 << "\", A_33 = \"" << A_33 << "\"" << std::endl;
        
        det = 10e10;
    }
    
    // calculate the inverse. Because the inverted matrix depends on other components of the matrix, 
    //  need to make a temporary value till all the inverted parts of the matrix are set
    double Ainv_11 =  (A_22*A_33 - A_23*A_32)/det;
    double Ainv_12 = -(A_12*A_33 - A_13*A_32)/det;
    double Ainv_13 =  (A_12*A_23 - A_22*A_13)/det;
    double Ainv_21 = -(A_21*A_33 - A_23*A_31)/det;
    double Ainv_22 =  (A_11*A_33 - A_13*A_31)/det;
    double Ainv_23 = -(A_11*A_23 - A_13*A_21)/det;
    double Ainv_31 =  (A_21*A_32 - A_31*A_22)/det;
    double Ainv_32 = -(A_11*A_32 - A_12*A_31)/det;
    double Ainv_33 =  (A_11*A_22 - A_12*A_21)/det;
    
    // now set the input reference A matrix to the temporary inverted A matrix values
    A_11 = Ainv_11;
    A_12 = Ainv_12;
    A_13 = Ainv_13;
    A_21 = Ainv_21;
    A_22 = Ainv_22;
    A_23 = Ainv_23;
    A_31 = Ainv_31;
    A_32 = Ainv_32;
    A_33 = Ainv_33;
    
}

void Plume::matmult(const double& A_11,const double& A_12,const double& A_13,
		            const double& A_21,const double& A_22,const double& A_23,
                    const double& A_31,const double& A_32,const double& A_33,
		            const double& b_11,const double& b_21,const double& b_31,
                    double& x_11, double& x_21, double& x_31)
{
    // since the x doesn't depend on itself, can just set the output without doing any temporary variables
    
    // now calculate the Ax=b x value from the input inverse A matrix and b matrix
    x_11 = b_11*A_11 + b_21*A_12 + b_31*A_13;
    x_21 = b_11*A_21 + b_21*A_22 + b_31*A_23;
    x_31 = b_11*A_31 + b_21*A_32 + b_31*A_33;
    
}


void Plume::setBCfunctions(std::string xBCtype,std::string yBCtype,std::string zBCtype)
{
    // the idea is to use the string input BCtype to determine which boundary condition function to use later in the program, and to have a function pointer
    // point to the required function. I learned about pointer functions from this website: https://www.learncpp.com/cpp-tutorial/78-function-pointers/
    
    // output some debug information
    if( debug == true ) {
        std::cout << "xBCtype = \"" << xBCtype << "\"" << std::endl;
        std::cout << "yBCtype = \"" << yBCtype << "\"" << std::endl;
        std::cout << "zBCtype = \"" << zBCtype << "\"" << std::endl;
    }
    
    if(xBCtype == "exiting") {
        // the enforceWallBCs_x pointer function now points to the enforceWallBCs_exiting function
        enforceWallBCs_x = &Plume::enforceWallBCs_exiting;  
    } else if(xBCtype == "periodic") {
        // the enforceWallBCs_x pointer function now points to the enforceWallBCs_periodic function
        enforceWallBCs_x = &Plume::enforceWallBCs_periodic;  
    } else if(xBCtype == "reflection") {
        // the enforceWallBCs_x pointer function now points to the enforceWallBCs_reflection function
        enforceWallBCs_x = &Plume::enforceWallBCs_reflection;  
    } else {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input xBCtype \"" << xBCtype 
                  << "\" has not been implemented in code! Available xBCtypes are "
                  << "\"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if(yBCtype == "exiting") {
        // the enforceWallBCs_y pointer function now points to the enforceWallBCs_exiting function
        enforceWallBCs_y = &Plume::enforceWallBCs_exiting;  
    } else if(yBCtype == "periodic") {
        // the enforceWallBCs_y pointer function now points to the enforceWallBCs_periodic function
        enforceWallBCs_y = &Plume::enforceWallBCs_periodic;  
    } else if(yBCtype == "reflection") {
        // the enforceWallBCs_y pointer function now points to the enforceWallBCs_reflection function
        enforceWallBCs_y = &Plume::enforceWallBCs_reflection;  
    } else {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input yBCtype \"" << yBCtype 
                  << "\" has not been implemented in code! Available yBCtypes are "
                  << "\"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if(zBCtype == "exiting") {
        // the enforceWallBCs_z pointer function now points to the enforceWallBCs_exiting function
        enforceWallBCs_z = &Plume::enforceWallBCs_exiting;  
    } else if(zBCtype == "periodic") {
        // the enforceWallBCs_z pointer function now points to the enforceWallBCs_periodic function
        enforceWallBCs_z = &Plume::enforceWallBCs_periodic;  
    } else if(zBCtype == "reflection") {
        // the enforceWallBCs_z pointer function now points to the enforceWallBCs_reflection function
        enforceWallBCs_z = &Plume::enforceWallBCs_reflection;  
    } else {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input zBCtype \"" << zBCtype 
                  << "\" has not been implemented in code! Available zBCtypes are "
                  << "\"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}



void Plume::enforceWallBCs_exiting(double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                   const double& domainStart,const double& domainEnd)
{
    // if it goes out of the domain, set isActive to false
    if( pos < domainStart || pos > domainEnd ) {
        isActive = false;
    }
}

void Plume::enforceWallBCs_periodic(double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                    const double& domainStart,const double& domainEnd)
{
    
    double domainSize = domainEnd - domainStart;
    
    /*    
          std::cout << "enforceWallBCs_periodic starting pos = \"" << pos << "\", domainStart = \"" << 
          domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;
    */
    
    if(domainSize != 0) {
        // before beginning of the domain => add domain length
        while( pos < domainStart ) {
            pos = pos + domainSize;
        }
        // past end of domain => sub domain length 
        while( pos > domainEnd ) {
            pos = pos - domainSize;
        }
    }
    
    /*
      std::cout << "enforceWallBCs_periodic ending pos = \"" << pos << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << std::endl;
    */
    
}

void Plume::enforceWallBCs_reflection(double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                      const double& domainStart,const double& domainEnd)
{
    if( isActive == true ) {
        
        /*
          std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velFluct = \"" << velFluct << "\", velFluct_old = \"" <<
          velFluct_old << "\", domainStart = \"" << domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;
        */
        
        int reflectCount = 0;
        while( (pos < domainStart || pos > domainEnd ) && reflectCount < 100) {
            // past end of domain or before beginning of the domain
            if( pos > domainEnd ) {
                pos = domainEnd - (pos - domainEnd);
                velFluct = -velFluct;
                velFluct_old = -velFluct_old;
            } else if( pos < domainStart ) {
                pos = domainStart - (pos - domainStart);
                velFluct = -velFluct;
                velFluct_old = -velFluct_old;
            }
            reflectCount = reflectCount + 1;
        }   // while outside of domain     
        
        // if the velocity is so large that the particle would reflect more than 100 times,
        // the boundary condition could fail.
        if (reflectCount == 100) {
            if( pos > domainEnd ) {
                std::cout << "warning (Plume::enforceWallBCs_reflection): "<<
                    "upper boundary condition failed! Setting isActive to false. pos = \"" << pos << "\"" << std::endl;
                isActive = false;
            } else if( pos < domainStart ) {
                std::cout << "warning (Plume::enforceWallBCs_reflection): "<<
                    "lower boundary condition failed! Setting isActive to false. xPos = \"" << pos << "\"" << std::endl;
                isActive = false;
            }
            
        }    
        /*
          std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velFluct = \"" << velFluct << "\", velFluct_old = \"" <<
          velFluct_old << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << loopCountRight << "\", reflectCount = \"" <<
          reflectCount << "\"" << std::endl;
        */
        
        
    }   // if isActive == true
}


void Plume::setFinishedParticleVals(double& xPos,double& yPos,double& zPos,bool& isActive,
                                    const bool& isRogue,
                                    const double& xPos_init, const double& yPos_init, const double& zPos_init)
{
    // FMargairaz -> this function will be used to removed particle from the particle list in the futur

    // need to set all rogue particles to inactive
    if( isRogue == true ) {
        isActive = false;
    }
    // now any inactive particles need set to the initial position
    // note: should we use the inital positio at inactive location for the particle?
    //if( isActive == false ) {
    //    xPos = xPos_init;
    //    yPos = yPos_init;
    //    zPos = zPos_init;
    //}
}

void Plume::writeSimInfoFile(const double& current_time)
{
    // if this output is not desired, skip outputting these files
    if( outputSimInfoFile == false )
    {
        return;
    }
    
    std::cout << "writing simInfoFile" << std::endl;
    
    
    // set some variables for use in the function
    FILE *fzout;    // changing file to which information will be written
    
    
    // now write out the simulation information to the debug folder
    
    // want a temporary variable to represent the caseBaseName, cause going to add on a timestep to the name
    std::string saveBasename = caseBaseName;
    
    // add timestep to saveBasename variable
    saveBasename = saveBasename + "_" + std::to_string(sim_dt);
    
    
    std::string outputFile = outputFolder + "/sim_info.txt";
    fzout = fopen(outputFile.c_str(), "w");
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"saveBasename     = %s\n",saveBasename.c_str());
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"C_0              = %lf\n",C_0);
    fprintf(fzout,"timestep         = %lf\n",sim_dt);
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"current_time     = %lf\n",current_time);
    fprintf(fzout,"rogueCount       = %0.0lf\n",isRogueCount);
    fprintf(fzout,"isNotActiveCount = %0.0lf\n",isNotActiveCount);
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"invarianceTol    = %lf\n",invarianceTol);
    fprintf(fzout,"velThreshold     = %lf\n",vel_threshold);
    fprintf(fzout,"\n");    // a purposeful blank line
    fclose(fzout);
    
    
    // now that all is finished, clean up the file pointer
    fzout = NULL;
    
}
