#include "Plume.hpp"

//void Plume::advectParticle(int& sim_tIdx, std::list<Particle*>::iterator parItr, WINDSGeneralData* WGD, TURBGeneralData* TGD, Eulerian* eul)
void Plume::advectParticle(double timeRemainder, std::list<Particle*>::iterator parItr, WINDSGeneralData* WGD, TURBGeneralData* TGD, Eulerian* eul)
{

    double rhoAir=1.225;   // in kg m^-3
    double nuAir=1.506E-5; // in m^2 s^-1
      
    // set settling velocity
    (*parItr)->setSettlingVelocity(rhoAir,nuAir);
    
    // get the current isRogue and isActive information
    bool isRogue = (*parItr)->isRogue;
    bool isActive = (*parItr)->isActive;
    
    // getting the current position for where the particle is at for a given time
    // if it is the first time a particle is ever released, then the value is already set at the initial value
    // LA notes: technically this value is the old position to be overwritten with the new position.
    //  I've been tempted for a while to store both. Might have to for correctly implementing reflective building BCs
    double xPos = (*parItr)->xPos;
    double yPos = (*parItr)->yPos;
    double zPos = (*parItr)->zPos;

    double uMean = 0.0;
    double vMean = 0.0;
    double wMean = 0.0;

    double flux_div_x = 0.0;
    double flux_div_y = 0.0;
    double flux_div_z = 0.0;

    //size_t cellIdx_old = eul->getCellId(xPos,yPos,zPos);
    
    // getting the initial position, for use in setting finished particles
    double xPos_init = (*parItr)->xPos_init;
    double yPos_init = (*parItr)->yPos_init;
    double zPos_init = (*parItr)->zPos_init;
    
    // grab the velFluct values.
    // LA notes: hmm, Bailey's code just starts out setting these values to zero,
    //  so the velFluct values are actually the old velFluct, that will be overwritten during the solver.
    //  velFluct_old and velFluct are probably identical and kind of redundant in this implementation
    //  but it shouldn't hurt anything for now, even if it is redundant
    //  besides, it will probably change a bit if we decide to change what is outputted on a regular, and on a debug basis
    double uFluct = (*parItr)->uFluct;
    double vFluct = (*parItr)->vFluct;
    double wFluct = (*parItr)->wFluct;
    
    // get all other values for the particle
    // in this case this, all the old velocity fluctuations and old stress tensor values for the particle
    // LA note: also need to keep track of a delta_velFluct, 
    //  but since delta_velFluct is never used, just set later on, it doesn't need grabbed as a value till later
    double uFluct_old = (*parItr)->uFluct_old;
    double vFluct_old = (*parItr)->vFluct_old;
    double wFluct_old = (*parItr)->wFluct_old;
    
    double txx_old = (*parItr)->txx_old;
    double txy_old = (*parItr)->txy_old;
    double txz_old = (*parItr)->txz_old;
    double tyy_old = (*parItr)->tyy_old;
    double tyz_old = (*parItr)->tyz_old;
    double tzz_old = (*parItr)->tzz_old;
    

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
    // FMargairaz -> need clean-up the comment

    while( isActive == true && timeRemainder > 0.0 ) {

        /*
          now get the Lagrangian values for the current iteration from the Eulerian grid
          will need to use the interp3D function
        */
                                        
        // set interoplation indexing variables for uFace variables
        eul->setInterp3Dindex_uFace(xPos,yPos,zPos);
        // interpolation of variables on uFace 
        uMean = eul->interp3D_faceVar(WGD->u);
        flux_div_x = eul->interp3D_faceVar(eul->dtxxdx);
        flux_div_y = eul->interp3D_faceVar(eul->dtxydx);
        flux_div_z = eul->interp3D_faceVar(eul->dtxzdx);
                    
        // set interpolation indexing variables for vFace variables
        eul->setInterp3Dindex_vFace(xPos,yPos,zPos);
        // interpolation of variables on vFace 
        vMean = eul->interp3D_faceVar(WGD->v);
        flux_div_x += eul->interp3D_faceVar(eul->dtxydy);
        flux_div_y += eul->interp3D_faceVar(eul->dtyydy);
        flux_div_z += eul->interp3D_faceVar(eul->dtyzdy);
                    
        // set interpolation indexing variables for wFace variables
        eul->setInterp3Dindex_wFace(xPos,yPos,zPos);
        // interpolation of variables on wFace 
        wMean = eul->interp3D_faceVar(WGD->w);
        flux_div_x += eul->interp3D_faceVar(eul->dtxzdz);
        flux_div_y += eul->interp3D_faceVar(eul->dtyzdz);
        flux_div_z += eul->interp3D_faceVar(eul->dtzzdz);

        // adjusting mean vertical velocity for settling velocity 
        wMean -= (*parItr)->vs;
        
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
        
        // this is the current reynolds stress tensor
        txx = eul->interp3D_cellVar(TGD->txx);
        txy = eul->interp3D_cellVar(TGD->txy);
        txz = eul->interp3D_cellVar(TGD->txz);
        tyy = eul->interp3D_cellVar(TGD->tyy);
        tyz = eul->interp3D_cellVar(TGD->tyz);
        tzz = eul->interp3D_cellVar(TGD->tzz);
        // now need to call makeRealizable on tau
        makeRealizable(txx,txy,txz,tyy,tyz,tzz);
        
        
        // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
        // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
        double par_dt = calcCourantTimestep(uMean+uFluct,vMean+vFluct,wMean+wFluct,timeRemainder);

        // update the par_time, useful for debugging
        //par_time = par_time + par_dt;
        
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
        isActive = invert3(lxx,lxy,lxz,lyx,lyy,lyz,lzx,lzy,lzz);
        if(isActive == false) {
            //int cellIdNew = eul->getCellId(xPos,yPos,zPos);    
            //std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
            //std::cerr << "PartID = " << (*parItr)->particleID << " in cell type: " << WGD->icellflag.at(cellIdNew) << std::endl;
            break;
        }
        // these are the random numbers for each direction
        // LA note: should be randn() matlab equivalent, which is a normally distributed random number
        // LA future work: it is possible the rogue particles are caused by the random number generator stuff.
        //  Need to look into it at some time.
        double xRandn = random::norRan();
        double yRandn = random::norRan();
        double zRandn = random::norRan();
        
        // now calculate a bunch of values for the current particle
        // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
        double dtxxdt = (txx - txx_old)/par_dt;
        double dtxydt = (txy - txy_old)/par_dt;
        double dtxzdt = (txz - txz_old)/par_dt;
        double dtyydt = (tyy - tyy_old)/par_dt;
        double dtyzdt = (tyz - tyz_old)/par_dt;
        double dtzzdt = (tzz - tzz_old)/par_dt;
                    
                    
        // now calculate and set the A and b matrices for an Ax = b 
        // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
        double A_11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*par_dt;
        double A_12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*par_dt;
        double A_13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*par_dt;
                    
        double A_21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*par_dt;
        double A_22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*par_dt;
        double A_23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*par_dt;
                    
        double A_31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*par_dt;
        double A_32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*par_dt;
        double A_33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*par_dt;
                    
        // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
        double b_11 = -uFluct_old - 0.50*flux_div_x*par_dt - std::sqrt(CoEps*par_dt)*xRandn;
        double b_21 = -vFluct_old - 0.50*flux_div_y*par_dt - std::sqrt(CoEps*par_dt)*yRandn;
        double b_31 = -wFluct_old - 0.50*flux_div_z*par_dt - std::sqrt(CoEps*par_dt)*zRandn;
                    
        // now prepare for the Ax=b calculation by calculating the inverted A matrix
        isActive = invert3(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33);
        if(isActive == false) {
            //std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
            break;
        }
        // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
        matmult(A_11,A_12,A_13,A_21,A_22,A_23,A_31,A_32,A_33,b_11,b_21,b_31, uFluct,vFluct,wFluct);
                    
                    
        // now check to see if the value is rogue or not
        if( std::abs(uFluct) >= vel_threshold || isnan(uFluct) ) {
            //std::cout << "Particle # " << (*parItr)->particleID << " is rogue, ";
            //std::cout << "responsible uFluct was \"" << uFluct << "\"" << std::endl;
            uFluct = 0.0;
            isActive = false;
            isRogue = true;
        }
        if( std::abs(vFluct) >= vel_threshold || isnan(vFluct) ) {
            //std::cerr << "Particle # " << (*parItr)->particleID << " is rogue, ";
            //std::cerr << "responsible vFluct was \"" << vFluct << "\"" << std::endl;
            vFluct = 0.0;
            isActive = false;
            isRogue = true;
        }
        if( std::abs(wFluct) >= vel_threshold || isnan(wFluct) ) {
            //std::cerr << "Particle # " << (*parItr)->particleID << " is rogue, ";
            //std::cerr << "responsible wFluct was \"" << wFluct << "\"" << std::endl;
            wFluct = 0.0;
            isActive = false;
            isRogue = true;
        }
                    
        // Pete: Do you need this???
        // ONLY if this should never happen....
        //    assert( isRogue == false );
        // LA reply: maybe implement this after the thesis work. Currently use it to know if something is going wrong
        //  I think we are still a long ways from being able to throw this out.
                    
        // now update the particle position for this iteration
        double disX = (uMean + uFluct)*par_dt;
        double disY = (vMean + vFluct)*par_dt;
        double disZ = (wMean + wFluct)*par_dt;
                    
        xPos = xPos + disX;
        yPos = yPos + disY;
        zPos = zPos + disZ;
        // check and do wall (building and terrain) reflection (based in the method)
        if( isActive == true ) isActive = (this->*wallReflection)(WGD,eul,xPos,yPos,zPos,disX,disY,disZ,uFluct,vFluct,wFluct);
            
                               
        // now apply boundary conditions
        if( isActive == true ) isActive = (this->*enforceWallBCs_x)(xPos,uFluct,uFluct_old,domainXstart,domainXend);
        if( isActive == true ) isActive = (this->*enforceWallBCs_y)(yPos,vFluct,vFluct_old,domainYstart,domainYend);
        if( isActive == true ) isActive = (this->*enforceWallBCs_z)(zPos,wFluct,wFluct_old,domainZstart,domainZend);
        
         
        // now set the particle values for if they are rogue or outside the domain
        // need to set all rogue particles to inactive
        if( isRogue == true ) {
            isActive = false;
        }
        // now any inactive particles need set to the initial position
        /* FMargairaz -> this has been deactivated
           if( isActive == false ) {
           xPos = xPos_init;
           yPos = yPos_init;
           zPos = zPos_init;
           }
        */
        
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
        
    }   // while( isActive == true && timeRemainder > 0.0 )
                
    // now update the old values and current values in the dispersion storage to be ready for the next iteration
    // also throw in the already calculated velFluct increment
    // notice that the values from the particle timestep loop are used directly here, 
    //  just need to put the existing vals into storage
    // !!! this is extremely important for output and the next iteration to work correctly
    (*parItr)->xPos = xPos;
    (*parItr)->yPos = yPos;
    (*parItr)->zPos = zPos;

    (*parItr)->uMean = uMean;
    (*parItr)->vMean = vMean;
    (*parItr)->wMean = wMean;
         
    (*parItr)->uFluct = uFluct;
    (*parItr)->vFluct = vFluct;
    (*parItr)->wFluct = wFluct;
                
    // these are the current velFluct values by this point
    (*parItr)->uFluct_old = uFluct_old;  
    (*parItr)->vFluct_old = vFluct_old;
    (*parItr)->wFluct_old = wFluct_old;
                
    (*parItr)->delta_uFluct = delta_uFluct;
    (*parItr)->delta_vFluct = delta_vFluct;
    (*parItr)->delta_wFluct = delta_wFluct;
                
    (*parItr)->txx_old = txx_old;
    (*parItr)->txy_old = txy_old;
    (*parItr)->txz_old = txz_old;
    (*parItr)->tyy_old = tyy_old;
    (*parItr)->tyz_old = tyz_old;
    (*parItr)->tzz_old = tzz_old;
                
    (*parItr)->isRogue = isRogue;
    (*parItr)->isActive = isActive;
    
    return;
}
