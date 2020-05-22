#include "Plume.hpp"

void Plume::advectParticle(int& sim_tIdx, int& parIdx, URBGeneralData* UGD, TURBGeneralData* TGD, Eulerian* eul, Dispersion* dis)
{
    
    // get the current isRogue and isActive information
    bool isRogue = dis->pointList[parIdx].isRogue;
    bool isActive = dis->pointList[parIdx].isActive;
    
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
    
    //size_t cellIdx_old = eul->getCellId(xPos,yPos,zPos);
    
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
                                        
        // set interoplation indexing variables for uFace variables
        eul->setInterp3Dindex_uFace(xPos,yPos,zPos);
        // interpolation of variables on uFace 
        double uMean = eul->interp3D_faceVar(UGD->u);
        double flux_div_x = eul->interp3D_faceVar(eul->dtxxdx);
        double flux_div_y = eul->interp3D_faceVar(eul->dtxydx);
        double flux_div_z = eul->interp3D_faceVar(eul->dtxzdx);
                    
        // set interpolation indexing variables for vFace variables
        eul->setInterp3Dindex_vFace(xPos,yPos,zPos);
        // interpolation of variables on vFace 
        double vMean = eul->interp3D_faceVar(UGD->v);
        flux_div_x += eul->interp3D_faceVar(eul->dtxydy);
        flux_div_y += eul->interp3D_faceVar(eul->dtyydy);
        flux_div_z += eul->interp3D_faceVar(eul->dtyzdy);
                    
        // set interpolation indexing variables for wFace variables
        eul->setInterp3Dindex_wFace(xPos,yPos,zPos);
        // interpolation of variables on wFace 
        double wMean = eul->interp3D_faceVar(UGD->w);
        flux_div_x += eul->interp3D_faceVar(eul->dtxzdz);
        flux_div_y += eul->interp3D_faceVar(eul->dtyzdz);
        flux_div_z += eul->interp3D_faceVar(eul->dtzzdz);
                    
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindex_cellVar(xPos,yPos,zPos);
                    
        // this is the Co times Eps for the particle
        // LA note: because Bailey's code uses Eps by itself and this does not, I wanted an option to switch between the two if necessary
        //  it's looking more and more like we will just use CoEps.
        double CoEps = eul->interp3D_cellVar(TGD->CoEps);
        // make sure CoEps is always bigger than zero
        if( CoEps <= 1e-6 ) {
            CoEps = 1e-6;
        }
        //double CoEps = eul->interp3D(turb->CoEps,"Eps");
        
        // this is the current reynolds stress tensor
        txx = eul->interp3D_cellVar(TGD->txx);
        txy = eul->interp3D_cellVar(TGD->txy);
        txz = eul->interp3D_cellVar(TGD->txz);
        tyy = eul->interp3D_cellVar(TGD->tyy);
        tyz = eul->interp3D_cellVar(TGD->tyz);
        tzz = eul->interp3D_cellVar(TGD->tzz);
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
        double lyx = txy;
        //double lyx = 0.0;
        double lyy = tyy;
        double lyz = tyz;
        double lzx = txz;
        double lzy = tyz;
        //double lzx = 0.0;
        //double lzy = 0.0;
        double lzz = tzz;
        invert3(lxx,lxy,lxz,lyx,lyy,lyz,lzx,lzy,lzz);
                    
                    
        // these are the random numbers for each direction
        // LA note: should be randn() matlab equivalent, which is a normally distributed random number
        // LA future work: it is possible the rogue particles are caused by the random number generator stuff.
        //  Need to look into it at some time.
        double xRandn = random::norRan();
        double yRandn = random::norRan();
        double zRandn = random::norRan();
        
        Vector3<double> vecRandn(random::norRan(), random::norRan(), random::norRan());
        
        /* now calculate a bunch of values for the current particle */
        // calculate the d_tau_dt values, which are the (tau_current - tau_old)/dt
        double dtxxdt = (txx - txx_old)/par_dt;
        double dtxydt = (txy - txy_old)/par_dt;
        double dtxzdt = (txz - txz_old)/par_dt;
        double dtyydt = (tyy - tyy_old)/par_dt;
        double dtyzdt = (tyz - tyz_old)/par_dt;
        double dtzzdt = (tzz - tzz_old)/par_dt;
                    
                    
        /* now calculate and set the A and b matrices for an Ax = b */
        // A = -I + 0.5*(-CoEps*L + L*dTdt)*par_dt;
        double A_11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*par_dt;
        double A_12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*par_dt;
        double A_13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*par_dt;
                    
        double A_21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*par_dt;
        double A_22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*par_dt;
        double A_23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*par_dt;
                    
        double A_31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*par_dt;
        double A_32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*par_dt;
        double A_33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*par_dt;
                    
        // b = -vectFluct - 0.5*vecFluxDiv*par_dt - std::sqrt(CoEps*par_dt)*vecRandn;
        double b_11 = -uFluct_old - 0.50*flux_div_x*par_dt - std::sqrt(CoEps*par_dt)*xRandn;
        double b_21 = -vFluct_old - 0.50*flux_div_y*par_dt - std::sqrt(CoEps*par_dt)*yRandn;
        double b_31 = -wFluct_old - 0.50*flux_div_z*par_dt - std::sqrt(CoEps*par_dt)*zRandn;
                    
        
        // A.invert()
        // vecFluct = A*b
        
        // now prepare for the Ax=b calculation by calculating the inverted A matrix
        invert3(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33);
        // now do the Ax=b calculation using the inverted matrix
        matmult(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33,b_11,b_21,b_31, uFluct,vFluct,wFluct);
                    
                    
        // now check to see if the value is rogue or not
        // if it is rogue, output a ton of information that can be copied into matlab
        // LA note: I tried to keep the format really nice to reduce the amount of reformulating work done in matlab.
        //  I wanted to turn it into a function, but there are sooo many variables that would need to be passed into that function call
        //  so it made more sense to write them out directly.
        if( ( std::abs(uFluct) >= dis->vel_threshold || isnan(uFluct) ) && nx > 1 ) {
            std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
            std::cout << "responsible uFluct was \"" << uFluct << "\"" << std::endl;
            uFluct = 0.0;
            isRogue = true;
        }
        if( ( std::abs(vFluct) >= dis->vel_threshold || isnan(vFluct) ) && ny > 1 ) {
            std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
            std::cout << "responsible vFluct was \"" << vFluct << "\"" << std::endl;
            vFluct = 0.0;
            isRogue = true;
        }
        if( ( std::abs(wFluct) >= dis->vel_threshold || isnan(wFluct) ) && nz > 1 ) {
            std::cout << "Particle # " << parIdx << " is rogue." << std::endl;
            std::cout << "responsible wFluct was \"" << wFluct << "\"" << std::endl;
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
        
        isActive = (this->*wallReflection)(UGD,eul,xPos,yPos,zPos,disX,disY,disZ,uFluct,vFluct,wFluct);
                                        
        // now apply boundary conditions
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
                    
        //cellIdx_old=cellIdx;
                    
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
                
    dis->pointList[parIdx].isRogue = isRogue;
    dis->pointList[parIdx].isActive = isActive;
    
    return;
}
