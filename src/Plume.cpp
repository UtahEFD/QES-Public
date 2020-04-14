//
//  Plume.cpp
//  
//  This class handles plume model
//

#include "Plume.hpp"

Plume::Plume( PlumeInputData* PID,Urb* urb,Dispersion* dis, Args* arguments) 
{
    
    std::cout<<"[Plume] \t Setting up simulation details "<<std::endl;
    
    // copy debug information
    doLagrDataOutput = arguments->doLagrDataOutput;
    outputSimInfoFile = arguments->doSimInfoFileOutput;
    outputFolder = arguments->outputFolder;
    caseBaseName = arguments->caseBaseName;
    debug = arguments->debug;

    // make local copies of the urb nVals for each dimension
    nx = urb->nx;
    ny = urb->ny;
    nz = urb->nz;
    dx = urb->dx;
    dy = urb->dy;
    dz = urb->dz;

    // get the domain start and end values, needed for wall boundary condition application
    domainXstart = dis->domainXstart;
    domainXend = dis->domainXend;
    domainYstart = dis->domainYstart;
    domainYend = dis->domainYend;
    domainZstart = dis->domainZstart;
    domainZend = dis->domainZend; 
    

    // make copies of important dispersion time variables
    sim_dt = dis->sim_dt;
    simDur = dis->simDur;
    nSimTimes = dis->nSimTimes;
    simTimes.resize(nSimTimes);   // first need to get the vector size right for the copy
    simTimes = dis->simTimes;

    // other important time variables not from dispersion
    CourantNum = PID->simParams->CourantNum;

    // make copy of dispersion number of particles to release at each simulation timestep
    // Note it is one less than times because the time loop ends one time early.
    nParsToRelease.resize(nSimTimes-1);    // first need to get the vector size right for the copy. 
    nParsToRelease = dis->nParsToRelease;

    
    // set additional values from the input
    invarianceTol = PID->simParams->invarianceTol;
    C_0 = PID->simParams->C_0;
    updateFrequency_timeLoop = PID->simParams->updateFrequency_timeLoop;
    updateFrequency_particleLoop = PID->simParams->updateFrequency_particleLoop;
    

    /* setup boundary condition functions */

    // now get the input boundary condition types from the inputs
    std::string xBCtype = PID->BCs->xBCtype;
    std::string yBCtype = PID->BCs->yBCtype;
    std::string zBCtype = PID->BCs->zBCtype;

    // now set the boundary condition function for the plume runs, 
    // and check to make sure the input BCtypes are legitimate
    setBCfunctions(xBCtype,yBCtype,zBCtype);

}

// LA note: in this whole section, the idea of having single value temporary storage instead of just referencing values
//  directly from the dispersion class seems a bit strange, but it makes the code easier to read cause smaller variable names.
//  Also, it is theoretically faster?
void Plume::run(Urb* urb,Turb* turb,Eulerian* eul,Dispersion* dis,std::vector<QESNetCDFOutput*> outputVec)
{
    std::cout << "[Plume] \t Advecting particles " << std::endl;

    // get the threshold velocity fluctuation to define rogue particles from dispersion class
    double vel_threshold = dis->vel_threshold;
    
    // //////////////////////////////////////////
    // TIME Stepping Loop
    // for every simulation time step
    // //////////////////////////////////////////
    
    
    if( debug == true )
    {
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
    dis->nParsReleased = 0;

    // want to output the particle information for the first timestep for where particles are without moving
    // so going to temporarily set nParsReleased to the number of particles released at the first time
    // do an output for the first time, then put the value back to zero so the particle loop will work correctly
    // this means I need to set the isActive value to true for the first set of particles right here
    dis->nParsReleased = nParsToRelease.at(0);
    for( int parIdx = 0; parIdx < nParsToRelease.at(0); parIdx++ ) {
        dis->pointList[parIdx].isActive = true;
    }
    for(size_t id_out=0;id_out<outputVec.size();id_out++) {
        outputVec.at(id_out)->save(simTimes.at(0));
    }
    dis->nParsReleased = 0;
    

    // LA note: that this loop goes from 0 to nTimes-2, not nTimes-1. This is because
    //  a given time iteration is calculating where particles for the current time end up for the next time
    //  so in essence each time iteration is calculating stuff for one timestep ahead of the loop.
    //  This also comes out in making sure the numPar to release makes sense. A list of times from 0 to 10 with timestep of 1
    //  means that nTimes is 11. So if the loop went from times 0 to 10 in that case, if 10 pars were released each time, 110 particles, not 100
    //  particles would end up being released.
    // LA note on debug timers: because the loop is doing stuff for the next time, and particles start getting released at time zero,
    //  this means that the updateFrequency needs to match with tStep+1, not tStep. At the same time, the current time output to consol
    //  output and to function calls need to also be set to tStep+1.
    for(int sim_tIdx = 0; sim_tIdx < nSimTimes-1; sim_tIdx++) {
     
        // need to release new particles
        // Add new particles to the number to move
        // !!! note that the updated number of particles is dispersion's value 
        //  so that the output can get the number of released particles right
        int nPastPars = dis->nParsReleased;
        dis->nParsReleased = nPastPars + nParsToRelease.at(sim_tIdx);

        // need to set the new particles isActive values to true
        for( int parIdx = nPastPars; parIdx < dis->nParsReleased; parIdx++ ) {
            dis->pointList[parIdx].isActive = true;
        }
        

        // only output the information when the updateFrequency allows and when there are actually released particles
        if(  nParsToRelease.at(sim_tIdx) != 0 && ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 )  )
        {
            std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) << "\". finished emitting \"" 
                      << nParsToRelease.at(sim_tIdx) << "\" particles from \"" << dis->allSources.size() 
                      << "\" sources. Total numParticles = \"" << dis->nParsReleased << "\"" << std::endl;
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
        int isRogueCount = dis->isRogueCount;
        int isNotActiveCount = dis->isNotActiveCount;

        for( int parIdx = 0; parIdx < dis->nParsReleased; parIdx++ ) {
            
            // get the current isRogue and isActive information
            bool isRogue = dis->pointList[parIdx].isRogue;
            bool isActive = dis->pointList[parIdx].isActive;


            // first check to see if the particle should even be advected and skip it if it should not be advected
            if( isActive == true ) {
                
                // get the amount of time it takes to advect a single particle, but only output the result when updateFrequency allows
                //  and when debugging
                if( debug == true ) {
                    if(  ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) 
                         && ( parIdx % updateFrequency_particleLoop == 0 || parIdx == dis->pointList.size()-1 )  ) {
                        // overall particle timer
                        timers.resetStoredTimer("particle iteration");
                    }
                }


                // getting the current position for where the particle is at for a given time
                // if it is the first time a particle is ever released, then the value is already set at the initial value
                // LA notes: technically this value is the old position to be overwritten with the new position.
                //  I've been tempted for a while to store both. Might have to for correctly implementing reflective building BCs
                double xPos = dis->pointList[parIdx].xPos;
                double yPos = dis->pointList[parIdx].yPos;
                double zPos = dis->pointList[parIdx].zPos;

                // getting the initial position, for use in setting finished particles
                double xPos_init = dis->pointList[parIdx].xPos_init;
                double yPos_init = dis->pointList[parIdx].yPos_init;
                double zPos_init = dis->pointList[parIdx].zPos_init;

                // grab the velFluct values.
                // LA notes: hmm, Bailey's code just starts out setting these values to zero,
                //  so the velFluct values are actually the old velFluct, that will be overwritten during the solver.
                //  velFluct_old and velFluct are probably identical and kind of redundant in this implementation
                //  but it shouldn't hurt anything for now, even if it is redundant
                //  besides, it will probably change a bit if we decide to change what is outputted on a regular, and on a debug basis
                double uFluct = dis->pointList[parIdx].uFluct;
                double vFluct = dis->pointList[parIdx].vFluct;
                double wFluct = dis->pointList[parIdx].wFluct;

                // get all other values for the particle
                // in this case this, all the old velocity fluctuations and old stress tensor values for the particle
                // LA note: also need to keep track of a delta_velFluct, 
                //  but since delta_velFluct is never used, just set later on, it doesn't need grabbed as a value till later
                double uFluct_old = dis->pointList[parIdx].uFluct_old;
                double vFluct_old = dis->pointList[parIdx].vFluct_old;
                double wFluct_old = dis->pointList[parIdx].wFluct_old;
                
                double txx_old = dis->pointList[parIdx].txx_old;
                double txy_old = dis->pointList[parIdx].txy_old;
                double txz_old = dis->pointList[parIdx].txz_old;
                double tyy_old = dis->pointList[parIdx].tyy_old;
                double tyz_old = dis->pointList[parIdx].tyz_old;
                double tzz_old = dis->pointList[parIdx].tzz_old;


                // need to avoid current tau values going out of scope now that I've added the particle timestep loop
                // so initialize their values to the tau_old values. They will be overwritten with the Eulerian grid value
                // at each iteration in the particle timestep loop
                double txx = txx_old;
                double txy = txy_old;
                double txz = txz_old;
                double tyy = tyy_old;
                double tyz = tyz_old;
                double tzz = tzz_old;

                // need to get the delta velFluct values right by doing the calculation inside the particle loop
                // these values go out of scope unless initialized here. So initialize them to zero (velFluct - velFluct_old = 0 right now)
                // they will be overwritten with the actual values in the particle timestep loop
                double delta_uFluct = 0.0;
                double delta_vFluct = 0.0;
                double delta_wFluct = 0.0;


                // time to do a particle timestep loop. start the time remainder as the simulation timestep.
                // at each particle timestep loop iteration the time remainder gets closer and closer to zero.
                // the particle timestep for a given particle timestep loop is either the time remainder or the value calculated
                // from the Courant Number, whichever is smaller.
                // particles can go inactive too, so need to use that as a condition to quit early too
                // LA important note: can't use the simulation timestep for the timestep remainder, the last simulation timestep
                //  is potentially smaller than the simulation timestep. So need to use the simTimes.at(nSimTimes-1)-simTimes.at(nSimTimes-2)
                //  for the last simulation timestep. The problem is that simTimes.at(nSimTimes-1) is greater than simTimes.at(nSimTimes-2) + sim_dt.
                double timeRemainder = sim_dt;
                if( sim_tIdx == nSimTimes-2 ) {  // at the final timestep
                    timeRemainder = simTimes.at(nSimTimes-1) - simTimes.at(nSimTimes-2);
                }
                double par_time = simTimes.at(sim_tIdx);    // the current time, updated in this loop with each new par_dt. 
                // Will end at simTimes.at(sim_tIdx+1) at the end of this particle loop
                
                while( isActive == true && timeRemainder > 0.0 ) {

                    // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
                    // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
                    double par_dt = calcCourantTimestep(uFluct,vFluct,wFluct,timeRemainder);

                    // update the par_time, useful for debugging
                    par_time = par_time + par_dt;

                    /*
                      now get the Lagrangian values for the current iteration from the Eulerian grid
                      will need to use the interp3D function
                    */


                    // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
                    // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
                    eul->setInterp3Dindexing(xPos,yPos,zPos);

                    // this is the Co times Eps for the particle
                    // LA note: because Bailey's code uses Eps by itself and this does not, I wanted an option to switch between the two if necessary
                    //  it's looking more and more like we will just use CoEps.
                    double CoEps = eul->interp3D(turb->CoEps);
                    // make sure CoEps is always bigger than zero
                    if( CoEps <= 1e-6 ) {
                        CoEps = 1e-6;
                    }
                    //double CoEps = eul->interp3D(turb->CoEps,"Eps");
                    
                    
                    // this is the current velMean value
                    double uMean = eul->interp3D(urb->u);
                    double vMean = eul->interp3D(urb->v);
                    double wMean = eul->interp3D(urb->w);
                    
                    // this is the current reynolds stress tensor
                    /* FM -> removed unnecessary copy
                    double txx_before = eul->interp3D(turb->txx,"tau");
                    double txy_before = eul->interp3D(turb->txy,"tau");
                    double txz_before = eul->interp3D(turb->txz,"tau");
                    double tyy_before = eul->interp3D(turb->tyy,"tau");
                    double tyz_before = eul->interp3D(turb->tyz,"tau");
                    double tzz_before = eul->interp3D(turb->tzz,"tau");
                    */

                    // now need flux_div_dir, not the different dtxxdx type components
                    double flux_div_x = eul->interp3D(eul->flux_div_x);
                    double flux_div_y = eul->interp3D(eul->flux_div_y);
                    double flux_div_z = eul->interp3D(eul->flux_div_z);


                    // now need to call makeRealizable on tau
                    // directly modifies the values of tau
                    // LA note: because the tau values before and after the function call are useful when particles go rogue,
                    //  I decided to store them separate using a copy for the function call
                    // note that these values are what is used to set the particle list values, they go out of scope if declared here
                    // so they are now declared outside the particle timestep iteration loop
                    /* FM -> removed unnecessary copy
                    txx = txx_before;
                    txy = txy_before;
                    txz = txz_before;
                    tyy = tyy_before;
                    tyz = tyz_before;
                    tzz = tzz_before;
                    */

                    // this is the current reynolds stress tensor
                    txx = eul->interp3D(turb->txx);
                    txy = eul->interp3D(turb->txy);
                    txz = eul->interp3D(turb->txz);
                    tyy = eul->interp3D(turb->tyy);
                    tyz = eul->interp3D(turb->tyz);
                    tzz = eul->interp3D(turb->tzz);
                    // now need to call makeRealizable on tau
                    makeRealizable(txx,txy,txz,tyy,tyz,tzz);
                    
                    
                    // now need to calculate the inverse values for tau
                    // directly modifies the values of tau
                    // LA warn: I just noticed that Bailey's code always leaves the last three components alone, 
                    //  never filled with the symmetrical tensor values. This seems fine for makeRealizable, 
                    //  but I wonder if it messes with the invert3 stuff since those values are used even though they are empty in his code
                    //  going to send in 9 terms anyways to try to follow Bailey's method for now
                    double lxx = txx;
                    double lxy = txy;
                    double lxz = txz;
                    //double lyx = txy;
                    double lyx = 0.0;
                    double lyy = tyy;
                    double lyz = tyz;
                    //double lzx = txz;
                    //double lzy = tyz;
                    double lzx = 0.0;
                    double lzy = 0.0;
                    double lzz = tzz;
                    invert3(lxx,lxy,lxz,lyx,lyy,lyz,lzx,lzy,lzz);
                    
                    
                    // these are the random numbers for each direction
                    // LA note: should be randn() matlab equivalent, which is a normally distributed random number
                    // LA future work: it is possible the rogue particles are caused by the random number generator stuff.
                    //  Need to look into it at some time.
                    double xRandn = random::norRan();
                    double yRandn = random::norRan();
                    double zRandn = random::norRan();
                    
                    
                    /* now calculate a bunch of values for the current particle */
                    // calculate the d_tau_dt values, which are the (tau_current - tau_old)/dt
                    double dtxxdt = (txx - txx_old)/par_dt;
                    double dtxydt = (txy - txy_old)/par_dt;
                    double dtxzdt = (txz - txz_old)/par_dt;
                    double dtyydt = (tyy - tyy_old)/par_dt;
                    double dtyzdt = (tyz - tyz_old)/par_dt;
                    double dtzzdt = (tzz - tzz_old)/par_dt;
                    
                    
                    /* now calculate and set the A and b matrices for an Ax = b */
                    double A_11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*par_dt;
                    double A_12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*par_dt;
                    double A_13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*par_dt;

                    double A_21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*par_dt;
                    double A_22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*par_dt;
                    double A_23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*par_dt;
                    
                    double A_31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*par_dt;
                    double A_32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*par_dt;
                    double A_33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*par_dt;


                    double b_11 = -uFluct_old - 0.50*flux_div_x*par_dt - std::sqrt(CoEps*par_dt)*xRandn;
                    double b_21 = -vFluct_old - 0.50*flux_div_y*par_dt - std::sqrt(CoEps*par_dt)*yRandn;
                    double b_31 = -wFluct_old - 0.50*flux_div_z*par_dt - std::sqrt(CoEps*par_dt)*zRandn;


                    /* FM -> removed unnecessary copy
                    // now prepare for the Ax=b calculation by calculating the inverted A matrix
                    // directly modifies the values of the A matrix
                    // LA note: because the A values before and after the function call are useful when particles go rogue,
                    //  I decided to store them separate using a copy for the function call
                    double A_11_inv = A_11;
                    double A_12_inv = A_12;
                    double A_13_inv = A_13;
                    double A_21_inv = A_21;
                    double A_22_inv = A_22;
                    double A_23_inv = A_23;
                    double A_31_inv = A_31;
                    double A_32_inv = A_32;
                    double A_33_inv = A_33;
                    invert3(A_11_inv,A_12_inv,A_13_inv,A_21_inv,A_22_inv,A_23_inv,A_31_inv,A_32_inv,A_33_inv);


                    // now do the Ax=b calculation using the inverted matrix
                    // directly modifies the velFluct values, which are passed in by reference as the output x vector
                    // LA note: since velFluct_old keeps track of the velFluct values before this function call,
                    //  I just used the velFluct values directly in the function call
                    matmult(A_11_inv,A_12_inv,A_13_inv,A_21_inv,A_22_inv,A_23_inv,A_31_inv,A_32_inv,A_33_inv,b_11,b_21,b_31, uFluct,vFluct,wFluct);
                    */
                    
                    // now prepare for the Ax=b calculation by calculating the inverted A matrix
                    invert3(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33);
                    // now do the Ax=b calculation using the inverted matrix
                    matmult(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33,b_11,b_21,b_31, uFluct,vFluct,wFluct);
                    

                    // now check to see if the value is rogue or not
                    // if it is rogue, output a ton of information that can be copied into matlab
                    // LA note: I tried to keep the format really nice to reduce the amount of reformulating work done in matlab.
                    //  I wanted to turn it into a function, but there are sooo many variables that would need to be passed into that function call
                    //  so it made more sense to write them out directly.
                    if( ( std::abs(uFluct) >= vel_threshold || isnan(uFluct) ) && nx > 1 ) {
                        std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
                        std::cout << "responsible uFluct was \"" << uFluct << "\"" << std::endl;
                        /*
                        std::cout << "\tinfo for matlab script copy:" << std::endl;
                        std::cout << "uFluct = " << uFluct << "\nvFluct = " << vFluct << "\nwFluct = " << wFluct << std::endl;
                        std::cout << "xPos = " << xPos << "\nyPos = " << yPos << "\nzPos = " << zPos << std::endl;
                        std::cout << "uFluct_old = " << uFluct_old << "\nvFluct_old = " << vFluct_old << "\nwFluct_old = " << wFluct_old << std::endl;
                        std::cout << "txx_old = " << txx_old << "\ntxy_old = " << txy_old << "\ntxz_old = " << txz_old << std::endl;
                        std::cout << "tyy_old = " << tyy_old << "\ntyz_old = " << tyz_old << "\ntzz_old = " << tzz_old << std::endl;
                        std::cout << "CoEps = " << CoEps << std::endl;
                        std::cout << "uMean = " << uMean << "\nvMean = " << vMean << "\nwMean = " << wMean << std::endl;
                        std::cout << "txx_before = " << txx_before << "\ntxy_before = " << txy_before << "\ntxz_before = " << txz_before << std::endl;
                        std::cout << "tyy_before = " << tyy_before << "\ntyz_before = " << tyz_before << "\ntzz_before = " << tzz_before << std::endl;
                        std::cout << "flux_div_x = " << flux_div_x << "\nflux_div_y = " << flux_div_y << "\nflux_div_z = " << flux_div_z << std::endl;
                        std::cout << "txx = " << txx << "\ntxy = " << txy << "\ntxz = " << txz << std::endl;
                        std::cout << "tyy = " << tyy << "\ntyz = " << tyz << "\ntzz = " << tzz << std::endl;
                        std::cout << "lxx = " << lxx << "\nlxy = " << lxy << "\nlxz = " << lxz << std::endl;
                        std::cout << "lyy = " << lyy << "\nlyz = " << lyz << "\nlzz = " << lzz << std::endl;
                        std::cout << "xRandn = " << xRandn << "\nyRandn = " << yRandn << "\nzRandn = " << zRandn << std::endl;
                        std::cout << "dtxxdt = " << dtxxdt << "\ndtxydt = " << dtxydt << "\ndtxzdt = " << dtxzdt << std::endl;
                        std::cout << "dtyydt = " << dtyydt << "\ndtyzdt = " << dtyzdt << "\ndtzzdt = " << dtzzdt << std::endl;
                        std::cout << "A_11 = " << A_11 << "\nA_12 = " << A_12 << "\nA_13 = " << A_13 << std::endl;
                        std::cout << "A_21 = " << A_21 << "\nA_22 = " << A_22 << "\nA_23 = " << A_23 << std::endl;
                        std::cout << "A_31 = " << A_31 << "\nA_32 = " << A_32 << "\nA_33 = " << A_33 << std::endl;
                        std::cout << "b_11 = " << b_11 << "\nb_21 = " << b_21 << "\nb_31 = " << b_31 << std::endl;
                        std::cout << "A_11_inv = " << A_11_inv << "\nA_12_inv = " << A_12_inv << "\nA_13_inv = " << A_13_inv << std::endl;
                        std::cout << "A_21_inv = " << A_21_inv << "\nA_22_inv = " << A_22_inv << "\nA_23_inv = " << A_23_inv << std::endl;
                        std::cout << "A_31_inv = " << A_31_inv << "\nA_32_inv = " << A_32_inv << "\nA_33_inv = " << A_33_inv << std::endl;
                        std::cout << "\t finished info" << std::endl;
                        */
                        uFluct = 0.0;
                        isRogue = true;
                    }
                    if( ( std::abs(vFluct) >= vel_threshold || isnan(vFluct) ) && ny > 1 ) {
                        std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
                        std::cout << "responsible vFluct was \"" << vFluct << "\"" << std::endl;
                        /*
                        std::cout << "\tinfo for matlab script copy:" << std::endl;
                        std::cout << "uFluct = " << uFluct << "\nvFluct = " << vFluct << "\nwFluct = " << wFluct << std::endl;
                        std::cout << "xPos = " << xPos << "\nyPos = " << yPos << "\nzPos = " << zPos << std::endl;
                        std::cout << "uFluct_old = " << uFluct_old << "\nvFluct_old = " << vFluct_old << "\nwFluct_old = " << wFluct_old << std::endl;
                        std::cout << "txx_old = " << txx_old << "\ntxy_old = " << txy_old << "\ntxz_old = " << txz_old << std::endl;
                        std::cout << "tyy_old = " << tyy_old << "\ntyz_old = " << tyz_old << "\ntzz_old = " << tzz_old << std::endl;
                        std::cout << "CoEps = " << CoEps << std::endl;
                        std::cout << "uMean = " << uMean << "\nvMean = " << vMean << "\nwMean = " << wMean << std::endl;
                        std::cout << "txx_before = " << txx_before << "\ntxy_before = " << txy_before << "\ntxz_before = " << txz_before << std::endl;
                        std::cout << "tyy_before = " << tyy_before << "\ntyz_before = " << tyz_before << "\ntzz_before = " << tzz_before << std::endl;
                        std::cout << "flux_div_x = " << flux_div_x << "\nflux_div_y = " << flux_div_y << "\nflux_div_z = " << flux_div_z << std::endl;
                        std::cout << "txx = " << txx << "\ntxy = " << txy << "\ntxz = " << txz << std::endl;
                        std::cout << "tyy = " << tyy << "\ntyz = " << tyz << "\ntzz = " << tzz << std::endl;
                        std::cout << "lxx = " << lxx << "\nlxy = " << lxy << "\nlxz = " << lxz << std::endl;
                        std::cout << "lyy = " << lyy << "\nlyz = " << lyz << "\nlzz = " << lzz << std::endl;
                        std::cout << "xRandn = " << xRandn << "\nyRandn = " << yRandn << "\nzRandn = " << zRandn << std::endl;
                        std::cout << "dtxxdt = " << dtxxdt << "\ndtxydt = " << dtxydt << "\ndtxzdt = " << dtxzdt << std::endl;
                        std::cout << "dtyydt = " << dtyydt << "\ndtyzdt = " << dtyzdt << "\ndtzzdt = " << dtzzdt << std::endl;
                        std::cout << "A_11 = " << A_11 << "\nA_12 = " << A_12 << "\nA_13 = " << A_13 << std::endl;
                        std::cout << "A_21 = " << A_21 << "\nA_22 = " << A_22 << "\nA_23 = " << A_23 << std::endl;
                        std::cout << "A_31 = " << A_31 << "\nA_32 = " << A_32 << "\nA_33 = " << A_33 << std::endl;
                        std::cout << "b_11 = " << b_11 << "\nb_21 = " << b_21 << "\nb_31 = " << b_31 << std::endl;
                        std::cout << "A_11_inv = " << A_11_inv << "\nA_12_inv = " << A_12_inv << "\nA_13_inv = " << A_13_inv << std::endl;
                        std::cout << "A_21_inv = " << A_21_inv << "\nA_22_inv = " << A_22_inv << "\nA_23_inv = " << A_23_inv << std::endl;
                        std::cout << "A_31_inv = " << A_31_inv << "\nA_32_inv = " << A_32_inv << "\nA_33_inv = " << A_33_inv << std::endl;
                        std::cout << "\t finished info" << std::endl;
                        */
                        vFluct = 0.0;
                        isRogue = true;
                    }
                    if( ( std::abs(wFluct) >= vel_threshold || isnan(wFluct) ) && nz > 1 ) {
                        std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
                        std::cout << "responsible wFluct was \"" << wFluct << "\"" << std::endl;
                        /*
                        std::cout << "\tinfo for matlab script copy:" << std::endl;
                        std::cout << "uFluct = " << uFluct << "\nvFluct = " << vFluct << "\nwFluct = " << wFluct << std::endl;
                        std::cout << "xPos = " << xPos << "\nyPos = " << yPos << "\nzPos = " << zPos << std::endl;
                        std::cout << "uFluct_old = " << uFluct_old << "\nvFluct_old = " << vFluct_old << "\nwFluct_old = " << wFluct_old << std::endl;
                        std::cout << "txx_old = " << txx_old << "\ntxy_old = " << txy_old << "\ntxz_old = " << txz_old << std::endl;
                        std::cout << "tyy_old = " << tyy_old << "\ntyz_old = " << tyz_old << "\ntzz_old = " << tzz_old << std::endl;
                        std::cout << "CoEps = " << CoEps << std::endl;
                        std::cout << "uMean = " << uMean << "\nvMean = " << vMean << "\nwMean = " << wMean << std::endl;
                        std::cout << "txx_before = " << txx_before << "\ntxy_before = " << txy_before << "\ntxz_before = " << txz_before << std::endl;
                        std::cout << "tyy_before = " << tyy_before << "\ntyz_before = " << tyz_before << "\ntzz_before = " << tzz_before << std::endl;
                        std::cout << "flux_div_x = " << flux_div_x << "\nflux_div_y = " << flux_div_y << "\nflux_div_z = " << flux_div_z << std::endl;
                        std::cout << "txx = " << txx << "\ntxy = " << txy << "\ntxz = " << txz << std::endl;
                        std::cout << "tyy = " << tyy << "\ntyz = " << tyz << "\ntzz = " << tzz << std::endl;
                        std::cout << "lxx = " << lxx << "\nlxy = " << lxy << "\nlxz = " << lxz << std::endl;
                        std::cout << "lyy = " << lyy << "\nlyz = " << lyz << "\nlzz = " << lzz << std::endl;
                        std::cout << "xRandn = " << xRandn << "\nyRandn = " << yRandn << "\nzRandn = " << zRandn << std::endl;
                        std::cout << "dtxxdt = " << dtxxdt << "\ndtxydt = " << dtxydt << "\ndtxzdt = " << dtxzdt << std::endl;
                        std::cout << "dtyydt = " << dtyydt << "\ndtyzdt = " << dtyzdt << "\ndtzzdt = " << dtzzdt << std::endl;
                        std::cout << "A_11 = " << A_11 << "\nA_12 = " << A_12 << "\nA_13 = " << A_13 << std::endl;
                        std::cout << "A_21 = " << A_21 << "\nA_22 = " << A_22 << "\nA_23 = " << A_23 << std::endl;
                        std::cout << "A_31 = " << A_31 << "\nA_32 = " << A_32 << "\nA_33 = " << A_33 << std::endl;
                        std::cout << "b_11 = " << b_11 << "\nb_21 = " << b_21 << "\nb_31 = " << b_31 << std::endl;
                        std::cout << "A_11_inv = " << A_11_inv << "\nA_12_inv = " << A_12_inv << "\nA_13_inv = " << A_13_inv << std::endl;
                        std::cout << "A_21_inv = " << A_21_inv << "\nA_22_inv = " << A_22_inv << "\nA_23_inv = " << A_23_inv << std::endl;
                        std::cout << "A_31_inv = " << A_31_inv << "\nA_32_inv = " << A_32_inv << "\nA_33_inv = " << A_33_inv << std::endl;
                        std::cout << "\t finished info" << std::endl;
                        */
                        wFluct = 0.0;
                        isRogue = true;
                    }

                    // Pete: Do you need this???
                    // ONLY if this should never happen....
                    //    assert( isRogue == false );
                    // LA reply: maybe implement this after the thesis work. Currently use it to know if something is going wrong
                    //  I think we are still a long ways from being able to throw this out.

                    
                    // now update the particle position for this iteration
                    // LA future work: at some point in time, need to do a CFL condition for only moving one eulerian grid cell at a time
                    //  this would mean adding some kind of while loop, with an adaptive timestep, controlling the end of the while loop
                    //   with the simulation time increment. This means all particles can do multiple time iterations, each with their own adaptive timestep.
                    //  To make this work, particles would need to be required to catch up so they are all calculated by a given simulation timestep.
                    // LA warn: currently, we use the simulation timestep so we may violate the eulerian grid CFL condition. 
                    //  but is simpler to work with when getting started
                    // LA future work: instead of using an adaptive timestep that adapts the timestep when checking if distX is too big or small,
                    //  we should use a courant number to precalculate the required adaptive timestep. Means less if statements and the error associated
                    //  with CFL conditions would just come out as the accuracy of the particle statistics.
                    double disX = (uMean + uFluct)*par_dt;
                    double disY = (vMean + vFluct)*par_dt;
                    double disZ = (wMean + wFluct)*par_dt;
                    
                    xPos = xPos + disX;
                    yPos = yPos + disY;
                    zPos = zPos + disZ;
                    
                    
                    // now apply boundary conditions
                    // LA note: notice that this is the old fashioned style for calling a pointer function
                    (this->*enforceWallBCs_x)(xPos,uFluct,uFluct_old,isActive, domainXstart,domainXend);
                    (this->*enforceWallBCs_y)(yPos,vFluct,vFluct_old,isActive, domainYstart,domainYend);
                    (this->*enforceWallBCs_z)(zPos,wFluct,wFluct_old,isActive, domainZstart,domainZend);
                    
                    
                    // now set the particle values for if they are rogue or outside the domain
                    setFinishedParticleVals(xPos,yPos,zPos,isActive, isRogue, xPos_init,yPos_init,zPos_init);
                    
                    // now update the old values to be ready for the next particle time iteration
                    // the current values are already set for the next iteration by the above calculations
                    // note: it may look strange to set the old values to the current values, then to use these
                    //  old values when setting the storage values, but that is what the old code was technically doing
                    //  we are already done using the old _old values by this point and need to use the current ones
                    // but we do need to set the delta velFluct values before setting the velFluct_old values to the current velFluct values
                    // !!! this is extremely important for the next iteration to work accurately
                    delta_uFluct = uFluct - uFluct_old;
                    delta_vFluct = vFluct - vFluct_old;
                    delta_wFluct = wFluct - wFluct_old;
                    uFluct_old = uFluct;
                    vFluct_old = vFluct;
                    wFluct_old = wFluct;
                    txx_old = txx;
                    txy_old = txy;
                    txz_old = txz;
                    tyy_old = tyy;
                    tyz_old = tyz;
                    tzz_old = tzz;


                    // now set the time remainder for the next loop
                    // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
                    // the function for calculating par_dt will use the timeRemainder for the output par_dt
                    // so this should result in a timeRemainder of exactly zero, no need for a tol.
                    timeRemainder = timeRemainder - par_dt;


                    // print info about the current particle time iteration
                    // but only if debug is set to true and this is the right updateFrequency time
                    if( debug == true ) {
                        if(  ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) 
                             && ( parIdx % updateFrequency_particleLoop == 0 || parIdx == dis->pointList.size()-1 )  ) {
                            std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) 
                                      << "\", par[" << parIdx << "]. Finished particle timestep \"" << par_dt 
                                      << "\" for time = \"" << par_time << "\"" << std::endl;
                        }
                    }
                    
                }   // while( isActive == true && timeRemainder > 0.0 )

                // now update the old values and current values in the dispersion storage to be ready for the next iteration
                // also throw in the already calculated velFluct increment
                // notice that the values from the particle timestep loop are used directly here, 
                //  just need to put the existing vals into storage
                // !!! this is extremely important for output and the next iteration to work correctly
                dis->pointList[parIdx].xPos = xPos;
                dis->pointList[parIdx].yPos = yPos;
                dis->pointList[parIdx].zPos = zPos;

                dis->pointList[parIdx].uFluct = uFluct;
                dis->pointList[parIdx].vFluct = vFluct;
                dis->pointList[parIdx].wFluct = wFluct;
                
                // these are the current velFluct values by this point
                dis->pointList[parIdx].uFluct_old = uFluct_old;  
                dis->pointList[parIdx].vFluct_old = vFluct_old;
                dis->pointList[parIdx].wFluct_old = wFluct_old;
                
                dis->pointList[parIdx].delta_uFluct = delta_uFluct;
                dis->pointList[parIdx].delta_vFluct = delta_vFluct;
                dis->pointList[parIdx].delta_wFluct = delta_wFluct;
                
                dis->pointList[parIdx].txx_old = txx_old;
                dis->pointList[parIdx].txy_old = txy_old;
                dis->pointList[parIdx].txz_old = txz_old;
                dis->pointList[parIdx].tyy_old = tyy_old;
                dis->pointList[parIdx].tyz_old = tyz_old;
                dis->pointList[parIdx].tzz_old = tzz_old;

                // now update the isRogueCount and isNotActiveCount
                if(isRogue == true) {
                    isRogueCount = isRogueCount + 1;
                }
                if(isActive == false) {
                    isNotActiveCount = isNotActiveCount + 1;
                }
                
                dis->pointList[parIdx].isRogue = isRogue;
                dis->pointList[parIdx].isActive = isActive;

                // get the amount of time it takes to advect a single particle, but only output the result when updateFrequency allows
                // LA future work: because this has timer related info, this probably needs to also be limited to when the user specifies they want debug mode
                if( debug == true ) {
                    if(  ( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 ) 
                         && ( parIdx % updateFrequency_particleLoop == 0 || parIdx == dis->pointList.size()-1 )  ) {
                        std::cout << "simTimes[" << sim_tIdx+1 << "] = \"" << simTimes.at(sim_tIdx+1) 
                                  << "\", par[" << parIdx << "]. finished particle iteration" << std::endl;
                        timers.printStoredTime("particle iteration");
                    }
                }
            
            }   // if isActive == true and isRogue == false


        } // for(int parIdx = 0; parIdx < dis->nParsReleased; parIdx++ )

        // set the isRogueCount and isNotActiveCount for the time iteration in the disperion data
        // !!! this needs set for the output to work properly
        dis->isRogueCount = isRogueCount;
        dis->isNotActiveCount = isNotActiveCount;


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
                      << dis->isRogueCount << "\", isNotActiveCount = \"" << dis->isNotActiveCount << "\"" << std::endl;
            // output advection loop runtime if in debug mode
            if( debug == true ) {
                timers.printStoredTime("advection loop");
            }
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


    // get the amount of time it takes to perform the simulation time integration loop
    // LA note: this is probably a debug output thing.
    if( debug == true ) {
        std::cout << "finished time integration loop" << std::endl;
        // Print out elapsed execution time
        timers.printStoredTime("simulation time integration loop");
    }


    // only outputs if the required booleans from input args are set
    // LA note: the current time put in here is one past when the simulation time loop ends
    //  this is because the loop always calculates info for one time ahead of the loop time.
    writeSimInfoFile(dis,simTimes.at(nSimTimes-1));

}


double Plume::calcCourantTimestep(const double& uFluct,const double& vFluct,const double& wFluct,const double& timeRemainder)
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
    double dt_x = CourantNum*dx/std::abs(uFluct);
    double dt_y = CourantNum*dy/std::abs(vFluct);
    double dt_z = CourantNum*dz/std::abs(wFluct);

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
    /*
      this may change as we figure out the reflections vs depositions on buildings and terrain as well as 
      the outer domain probably will become some kind of inherited function or a pointer function that can 
      be chosen at initialization time for now, if it goes out of the domain, set isActive to false
    */
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

void Plume::writeSimInfoFile(Dispersion* dis, const double& current_time)
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
    fprintf(fzout,"rogueCount       = %0.0lf\n",dis->isRogueCount);
    fprintf(fzout,"isNotActiveCount = %0.0lf\n",dis->isNotActiveCount);
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"invarianceTol    = %lf\n",invarianceTol);
    fprintf(fzout,"velThreshold     = %lf\n",dis->vel_threshold);
    fprintf(fzout,"\n");    // a purposeful blank line
    fclose(fzout);


    // now that all is finished, clean up the file pointer
    fzout = NULL;

}
