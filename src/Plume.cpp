//
//  Plume.cpp
//  
//  This class handles plume model
//

#include "Plume.hpp"

Plume::Plume(Urb* urb,Dispersion* dis, PlumeInputData* PID, Output* output) {
    
    std::cout<<"[Plume] \t Setting up particles "<<std::endl;
    
    Wind windRot;   // strange, this doesn't appear to be used in this function, so it is an unnecessary declaration
    
    // make local copies
    nx = urb->grid.nx;
    ny = urb->grid.ny;
    nz = urb->grid.nz;
    
    /* setup the sampling box concentration information */

    numPar = PID->sources->numParticles;
    
    nBoxesX = PID->colParams->nBoxesX;
    nBoxesY = PID->colParams->nBoxesY;
    nBoxesZ = PID->colParams->nBoxesZ;
    
    boxSizeX = (PID->colParams->boxBoundsX2-PID->colParams->boxBoundsX1)/nBoxesX;	  
    boxSizeY = (PID->colParams->boxBoundsY2-PID->colParams->boxBoundsY1)/nBoxesY;	  
    boxSizeZ = (PID->colParams->boxBoundsZ2-PID->colParams->boxBoundsZ1)/nBoxesZ;
    
    volume=boxSizeX*boxSizeY*boxSizeZ;
    
    lBndx = PID->colParams->boxBoundsX1;
    uBndx = PID->colParams->boxBoundsX2;
    lBndy = PID->colParams->boxBoundsY1;
    uBndy = PID->colParams->boxBoundsY2;
    lBndz = PID->colParams->boxBoundsZ1;
    uBndz = PID->colParams->boxBoundsZ2;
    
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);
    zBoxCen.resize(nBoxesZ);
    
    quanX = (uBndx-lBndx)/(nBoxesX);
    quanY = (uBndy-lBndy)/(nBoxesY);
    quanZ = (uBndz-lBndz)/(nBoxesZ);
    
    int id=0;   // this isn't used apparently
    int zR=0;
    int yR=0;
    int xR=0;
    for(int k=0;k<nBoxesZ;++k) {
        zBoxCen.at(k) = lBndz + (zR*quanZ) + (boxSizeZ/2.0);        // the .at(k) is different from [k] in that it checks the bounds, throwing an out of range error if outside the bounds of the vector
        zR++;
    }
    for(int j=0;j<nBoxesY;++j) {
        yBoxCen.at(j) = lBndy + (yR*quanY) + (boxSizeY/2.0);
        yR++;
    }
    for(int i=0;i<nBoxesX;++i) {
        xBoxCen.at(i) = lBndx + (xR*quanX) + (boxSizeX/2.0);
        xR++;
    }
    
    /* make copies of important dispersion information for particle release */

    tStepInp = PID->simParams->timeStep;
    avgTime  = PID->colParams->timeAvg;
    
    sCBoxTime = PID->colParams->timeStart;
    numTimeStep  = dis->numTimeStep;
    
    tStrt.resize(numPar);
    tStrt = dis->tStrt;
    
    timeStepStamp.resize(numTimeStep);
    timeStepStamp = dis->timeStepStamp;
    
    parPerTimestep=dis->parPerTimestep;
    
    cBox.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    conc.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    
    /* setup output information */

    // set cell-centered dimensions
    NcDim t_dim = output->addDimension("t");
    NcDim z_dim = output->addDimension("z",nBoxesZ);
    NcDim y_dim = output->addDimension("y",nBoxesY);
    NcDim x_dim = output->addDimension("x",nBoxesX);

    dim_scalar_t.push_back(t_dim);
    dim_scalar_z.push_back(z_dim);
    dim_scalar_y.push_back(y_dim);
    dim_scalar_x.push_back(x_dim);
    
    dim_vector.push_back(t_dim);
    dim_vector.push_back(z_dim);
    dim_vector.push_back(y_dim);
    dim_vector.push_back(x_dim);

    // create attributes
    AttScalarDbl att_t     = {&timeOut, "t",    "time",        "s",  dim_scalar_t};
    AttVectorDbl att_x     = {&xBoxCen, "x",    "x-distance",  "m",  dim_scalar_x};
    AttVectorDbl att_y     = {&yBoxCen, "y",    "y-distance",  "m",  dim_scalar_y};
    AttVectorDbl att_z     = {&zBoxCen, "z",    "z-distance",  "m",  dim_scalar_z};
    AttVectorDbl att_conc  = {&conc,    "conc", "concentratio","--", dim_vector};

    // map the name to attributes
    map_att_scalar_dbl.emplace("t", att_t);
    map_att_vector_dbl.emplace("x", att_x);
    map_att_vector_dbl.emplace("y", att_y);
    map_att_vector_dbl.emplace("z", att_z);
    map_att_vector_dbl.emplace("conc", att_conc);
    
    // stage fields for output
    output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);
    output_vector_dbl.push_back(map_att_vector_dbl["x"]);
    output_vector_dbl.push_back(map_att_vector_dbl["y"]);
    output_vector_dbl.push_back(map_att_vector_dbl["z"]);
    output_vector_dbl.push_back(map_att_vector_dbl["conc"]);

    // add scalar double fields
    for ( AttScalarDbl att : output_scalar_dbl ) {
        output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }

    // add vector double fields
    for ( AttVectorDbl att : output_vector_dbl ) {
        output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }
}

// This is here only so we still have the old and ugly method to look at. As soon as I have the better stuff done, I'm throwing this out!
#define USE_NEWCODE 1

#if USE_NEWCODE
void Plume::run(Urb* urb, Turb* turb, Eulerian* eul, Dispersion* dis, PlumeInputData* PID, Output* output)
{
    
    std::cout<<"[Plume] \t Advecting particles "<<std::endl;

    // set some variables before the time integration
    int parToMove=0;    // this is the loop counter ending point for the particle loop. Looks like it gets adjusted each time to add in the extra particles

    double dt=tStepInp;     // this is a cleaner name for the timestep

    // need to calculate the velFluct threshold value for rogue particles
    // the threshold velocity fluctuation to define rogue particles
    double vel_threshold = dis->vel_threshold;
    
    // For every time step
    for(tStep=0; tStep<numTimeStep; tStep++) {
        
        // Move each particle for every time step
        parToMove = parToMove + parPerTimestep;     // add the new particles to the total number to be moved each timestep
        double isRogueCount = dis->isRogueCount;

        for(int par=0; par<parToMove;par++) {

            // first check to see if the particle should even be advected
            bool isRogue = dis->isRogue.at(par);
            bool isActive = dis->isActive.at(par);

            if(isActive == true && isRogue == false)
            {

                // this is getting the current position for where the particle is at for a given time
                // if it is the first time a particle is ever released, then the value is already set at the initial value
                double xPos = dis->pos.at(par).e11;
                double yPos = dis->pos.at(par).e21;
                double zPos = dis->pos.at(par).e31;

                // this is the old velFluct value
                double uPrime = dis->prime.at(par).e11;
                double vPrime = dis->prime.at(par).e21;
                double wPrime = dis->prime.at(par).e31;

                // should also probably grab and store the old values in this same way
                // these consist of velFluct_old and tao_old
                // also need to keep track of a delta_velFluct and an isActive flag for each particle
                // though delta_velFluct doesn't need grabbed as a value till later now that I think on it
                double uFluct_old = dis->prime_old.at(par).e11;
                double vFluct_old = dis->prime_old.at(par).e21;
                double wFluct_old = dis->prime_old.at(par).e31;
                double txx_old = dis->tau_old.at(par).e11;
                double txy_old = dis->tau_old.at(par).e12;
                double txz_old = dis->tau_old.at(par).e13;
                double tyy_old = dis->tau_old.at(par).e22;
                double tyz_old = dis->tau_old.at(par).e23;
                double tzz_old = dis->tau_old.at(par).e33;
                
                
                /*
                    now get the values for the current iteration
                    will need to use the interp3D function
                    Need to get velMean, CoEps, tao, flux_div_vel
                    Then need to call makeRealizable on tao then calculate inverse tao
                    
                    Okay should I use the given inverse tao from Turb or no? I'm going to go with no for now

                    I guess for now, I can just use the old interpolation method
                */


                // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
                // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
                eul->setInterp3Dindexing(pos.at(par));



                // this is the Co times Eps for the particle
                double CoEps = eul->interp3D(turb->CoEps);
                
                
                // this is the current velMean value
                Wind velMean = eul->interp3D(urb->wind);
                double uMean = velMean.u;
                double vMean = velMean.v;
                double wMean = velMean.w;
                
                matrix6 tao = eul->interp3D(turb->tau);
                double txx = tao.e11;
                double txy = tao.e12;
                double txz = tao.e13;
                double tyy = tao.e21;
                double tyz = tao.e22;
                double tzz = tao.e23;
                
                // now need flux_div_vel not the different dtxxdx type components
                vec3 flux_div_vel = eul->interp3D(eul->flux_div);
                double flux_div_u = flux_div_vel.e11;
                double flux_div_v = flux_div_vel.e12;
                double flux_div_w = flux_div_vel.e13;


                // now need to call makeRealizable on tao
                //makeRealizable();

                // now need to calculate the inverse values for tao
                // might be able to get away with currently existing functions for this. May still be good to write my own
                // again already existing function Eulerian->matrixInv() for matrix6 to matrix 9 seems close to what I want
                // I'm probably going to write my own function, but it looks like using the structs might be the best way to control output
                // either that or passing in the variables by reference that need changed



                // these are the random numbers for each direction
                double xRandn = random::norRan();
                double yRandn = random::norRan();
                double zRandn = random::norRan();


                
                /* now calculate a bunch of values for the current particle */
                // calculate the d_tao_dt values, which are the (tao_current - tao_old)/dt
                double dtxxdt = (txx - txx_old)/dt;
                double dtxydt = (txy - txy_old)/dt;
                double dtxzdt = (txz - txz_old)/dt;
                double dtyydt = (tyy - tyy_old)/dt;
                double dtyzdt = (tyz - tyz_old)/dt;
                double dtzzdt = (tzz - tzz_old)/dt;


                /* now calculate and set the A and b matrices for an Ax = b */
                double A[3][3] = {0};
                double b[3] = {0};

                A[1][1] = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*dt;
                A[1][2] =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*dt;
                A[1][3] =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*dt;

                A[2][1] =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*dt;
                A[2][2] = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*dt;
                A[2][3] =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*dt;
                
                A[3][1] =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*dt;
                A[3][2] =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*dt;
                A[3][3] = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*dt;


                b[1] = -uFluct_old - 0.50*flux_div_x*dt - sqrt(CoEps*dt)*xRandn;
                b[2] = -vFluct_old - 0.50*flux_div_y*dt - sqrt(CoEps*dt)*yRandn;
                b[3] = -wFluct_old - 0.50*flux_div_z*dt - sqrt(CoEps*dt)*zRandn;


                // now prepare for the Ax=b calculation by calculating the inverted A matrix
                // need to prepare the invert function, mayone one already exists in Eulerian since they do an Ax=b calculation
                // not sure if these inputs and outputs are allowable or make sense either. Seems off somehow to me.
                // the Eulerian->matrixInv() function for inverting a matrix6 to become a matrix9 looks closest to Brian's matrix inversion function
                // but there appears to be some differences in the sign stuff, and how to handle if the determinant is too small
                // the other Eulerian->matrixInv() is for matrix9 in to matrix9 out, but has a bunch of extra stuff that looks kind of like parts of makeRealizable

                double Ainv[3][3] = invert3(A);


                // now do the Ax=b calculation using the inverted matrix
                // need to either write this function, or find it in the Eulerian Ax=b stuff
                // hm, since getting out three values, might be easier to do a pass in by reference thing for the velPrime values
                // Brian's matmult looks like Eulerian->matrixVecMult, so I guess if the datatypes are right, could use that function
                matmult(Ainv,b,  uPrime,vPrime,wPrime);
                

                // now check to see if the value is rogue or not
                if( abs(uPrime) >= vel_threshold || isnan(uPrime) && nx > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue.\n";
                    std::cout << "responsible uFluct was \"" << uPrime << "\"\n";
                    uPrime = 0.0;
                    isRogue = true;
                }
                if( abs(vPrime) >= vel_threshold || isnan(vPrime) && ny > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue.\n";
                    std::cout << "responsible vFluct was \"" << vPrime << "\"\n";
                    vPrime = 0.0;
                    isRogue = true;
                }
                if( abs(wPrime) >= vel_threshold || isnan(wPrime) && nz > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue.\n";
                    std::cout << "responsible wFluct was \"" << wPrime << "\"\n";
                    wPrime = 0.0;
                    isRogue = true;
                }
                
                // now update the particle position for this iteration
                double disX = ((uMean + uPrime)*dt);
                double disY = ((vMean + vPrime)*dt);
                double disZ = ((wMean + wPrime)*dt);
                
                xPos = xPos + disX;
                yPos = yPos + disY;
                zPos = zPos + disZ;


                // now apply boundary conditions
                // we may want to skip this one, or maybe just set the particle to being inactive if it reaches outside the domain
                // the reason for this, is Brian's BCs don't match what is normally used for Plume, we don't want reflections or periodic BCs
                // when we allow different test cases, we will want these options, and a way to choose the boundary condition type
                // for different regions sometime during the constructor phases.
                // I guess just implement one that makes isActive go false if it goes outside the domain
                //enforceWallBCs();


                // now update the old values and current values in the dispersion storage to be ready for the next iteration
                // also calculate the velFluct increment
                dis->prime.at(par).e11 = uPrime;
                dis->prime.at(par).e21 = vPrime;
                dis->prime.at(par).e31 = wPrime;
                dis->pos.at(par).e11 = xPos;
                dis->pos.at(par).e21 = yPos;
                dis->pos.at(par).e31 = zPos;

                dis->delta_prime.at(par).e11 = uPrime - uFluct_old;
                dis->delta_prime.at(par).e21 = vPrime - vFluct_old;
                dis->delta_prime.at(par).e31 = wPrime - wFluct_old;
                dis->prime_old.at(par).e11 = uPrime;
                dis->prime_old.at(par).e21 = vPrime;
                dis->prime_old.at(par).e31 = wPrime;

                dis->tau_old.at(par).e11 = txx;
                dis->tau_old.at(par).e12 = txy;
                dis->tau_old.at(par).e13 = txz;
                dis->tau_old.at(par).e22 = tyy;
                dis->tau_old.at(par).e23 = tyz;
                dis->tau_old.at(par).e33 = tzz;

                if(isRogue == true)
                {
                    isRogueCount = isRogueCount + 1;
                }

                dis->isRogue.at(par) = isRogue;
                dis->isActive.at(par) = isActive;
            
            }   // if isActive == true and isRogue == false

        } // for (int par=0; par<parToMove;par++)

        // set the isRogueCount for the time iteration in the disperion data
        // hm, I'm almost wondering if this needs to go into dispersion, could just be kept locally,
        // but declared outside the loop to preserve the value
        dis->isRogueCount = isRogueCount;

        
        // this is basically saying, if we are past the time to start averaging values to calculate the concentration,
        // then calculate the average, where average does . . .
        if(timeStepStamp.at(tStep) >= sCBoxTime){
            average(tStep,dis,urb);
        }
        // this is basically saying, if the current time has passed a time that we should be outputting values at,
        // then calculate the concentrations for the sampling boxes, save the output for this timestep, and update time counter for when to do it again.
        // I'm honestly confused why cBox gets set to zero, unless this is meant to cleanup for the next iteration. If this is so, then why do it in a way
        // that you can't output the information if you ever need to debug it? Seems like it should be a temporary variable then.
        if(timeStepStamp.at(tStep)>= sCBoxTime+avgTime) {
            //std::cout<<"loopPrm   :"<<loopPrm<<std::endl;
            //std::cout<<"loopLowestCell :"<<loopLowestCell<<std::endl;
            double cc=(tStepInp)/(avgTime*volume*numPar);
            for(int k=0;k<nBoxesZ;k++) {
                for(int j=0;j<nBoxesY;j++) {
                    for(int i=0;i<nBoxesX;i++) {
                        int id=k*nBoxesY*nBoxesX+j*nBoxesX+i;
                        conc.at(id) = cBox.at(id)*cc;
                        cBox.at(id) = 0.0;
                    }
                }
            }
            save(output);
            avgTime=avgTime+PID->colParams->timeAvg;    // I think this is updating the averaging time for the next loop
        }

    } // for(tStep=0; tStep<numTimeStep; tStep++)

}

matrix6 Plume::makeRealizable(const matrix6& tau)
{
    // need to fill out this function. Is going to be some work cause now the data structures are wierd
}

void enforceBCs(double&)
{
    // need to fill out this function. Is going to be some work cause now the data structures are wierd, and I can't output multiple values at once
    // this one may also change as we figure out the reflections vs depositions on buildings and terrain as well as the outer domain
    // probably will become some kind of inherited function or a pointer function that can be chosen at initialization time
}


#else   // USE_NEWCODE == 0
void Plume::run(Urb* urb, Turb* turb, Eulerian* eul, Dispersion* dis, PlumeInputData* PID, Output* output) {
    
    std::cout<<"[Plume] \t Advecting particles "<<std::endl;
    
    // these flags are either to warn of a problem, or to control how the iteration goes and what methods to use for the iteration during each iteration
    // the counts appear to be related to the flags, some kind of way to also control the number of iterations
    int Flag=0;     // this is just to warn/flag if the zPos becomes less than the zo roughness length found in the euler stuff. Yet it is unused other than being set to 1
    int flag=0;     // this is not used. Took me a bit to track it down
    int flagPrime=0;    // some kind of check on some value stuff in the calculation
    int loopPrm=0;      // this is just a count for the number of particles per timestep that flagPrime gets set to 1
    int loopLowestCell=0;   // not used
    int flag_g2nd=0;        // some kind of check on some value stuff in the calculation
    int countMax=10;//100;
    int countPrmMax=10;//1000;
    double ranU=0.0;    // storage container for a random number in the U direction for each iteration. Looks like it is a normalized random number
    double ranV=0.0;
    double ranW=0.0;
    int parToMove=0;    // this is the loop counter ending point for the particle loop. Looks like it gets adjusted each time to add in the extra particles
    
    // For every time step
    for(tStep=0; tStep<numTimeStep; tStep++) {
        
        // Move each particle for every time step
        parToMove = parToMove + parPerTimestep;     // add the new particles to the total number to be moved each timestep
        
        for(int par=0; par<parToMove;par++) {
            loopPrm=0;      // this is just a count for the number of particles per timestep that flagPrime gets set to 1
            
            int count=0;
            int countPrm=0;
            double xPos = dis->pos.at(par).x;   // this is getting the current position for where the particle is at for a given time
            double yPos = dis->pos.at(par).y;
            double zPos = dis->pos.at(par).z;
                        
            double tStepRem=tStepInp;   // I guess the dt isn't a good variable name or can't be used again?. This is the remaining part of the timestep still to be stepped through.
            double tStepUsed=0.0;       // this is the actual timestep that was used for a given timestep iteration
            double tStepCal=0.0;        // still confused by this, but it appears to be some method for adapting the timestep, taking a minimum of the different times that can vary, and dividing again
            double dt=tStepRem;     // isn't this just the dt again? why not use the variable you've set in the class? I guess it is an adaptive timestepping thing. Okay now I see, it is the actual dt that is varying as the timestep is adapted a lot. So this is the adapting timestep, not necessarily the final used timestep till the end
            int loops=0;        // appears to be a counter variable for how many loops happen, but isn't really used
            double tStepMin=tStepInp;   // another form of the adaptive timestepping stuff again! Looks like the minimum it can be and is used to force the time to shrink for the next iteration
            int loopTby2=0;     // another counter variable, how many times does it loop over the timestep in it's adaptive method. But isn't used anywhere
            
            while(tStepRem>1.0e-5) {        // looks like an iterative process till something converges
                // a set of grid indexing variables
                int iV=int(xPos/urb->grid.dx);
                int jV=int(yPos/urb->grid.dy);
                int kV=int(zPos/urb->grid.dz)+1;
                int id=kV*ny*nx+jV*nx+iV;
                
                // if the partice is in domain and ready to be released*/
                // so this is a way to ignore particles that go bad without restarting them
                // is it in the domain? and is the wind velocity not zero for the closest cell?
                // hm, so this is eulerian cell grid ID stuff
                if(iV>0 && iV<nx-1 && jV>0 && jV<ny-1 && kV>0 && kV<nz-1 && urb->grid.icell.at(id)!=0) {
                        
	                loops++;                
                    
                    /* looks like the first thing is to get a bunch of values for the current particle */

                    double eigVal_11=eul->eigVal.at(id).e11;
                    double eigVal_22=eul->eigVal.at(id).e22;
                    double eigVal_33=eul->eigVal.at(id).e33;
                    
                    double CoEps=turb->CoEps.at(id);
                    double tFac=0.5;
                    double tStepSigW=(2.0*(turb->sig.at(id).e33)*(turb->sig.at(id).e33)/CoEps);
                    double tStepEig11=-1.0/eigVal_11;
                    double tStepEig22=-1.0/eigVal_22;
                    double tStepEig33=-1.0/eigVal_33;
                    
                    double tStepArr[]={fabs(tStepEig11),fabs(tStepEig22),fabs(tStepEig33),fabs(tStepSigW)};
                    tStepCal=tFac * min(tStepArr,4); 
                    double arrT[]={tStepMin,tStepCal,tStepRem,dt};
                    dt=min(arrT,4);
                    double uPrime=dis->prime.at(par).x;     // this is the current velFluct value not the position!
                    double vPrime=dis->prime.at(par).y;
                    double wPrime=dis->prime.at(par).z;
                    double uMean=urb->wind.at(id).u;
                    double vMean=urb->wind.at(id).v;
                    double wMean=urb->wind.at(id).w;
                    double ka0_11=eul->ka0.at(id).e11;
                    double ka0_21=eul->ka0.at(id).e21;
                    double ka0_31=eul->ka0.at(id).e31;
                    double g2nd_11=eul->g2nd.at(id).e11;
                    double g2nd_21=eul->g2nd.at(id).e21;
                    double g2nd_31=eul->g2nd.at(id).e31;
                    
                    double lam11=turb->lam.at(id).e11;
                    double lam12=turb->lam.at(id).e12;
                    double lam13=turb->lam.at(id).e13;
                    double lam21=turb->lam.at(id).e21;
                    double lam22=turb->lam.at(id).e22;
                    double lam23=turb->lam.at(id).e23;
                    double lam31=turb->lam.at(id).e31;
                    double lam32=turb->lam.at(id).e32;
                    double lam33=turb->lam.at(id).e33;
                    
                    double taudx11=eul->taudx.at(id).e11;
                    double taudx12=eul->taudx.at(id).e12;
                    double taudx13=eul->taudx.at(id).e13;
                    double taudx22=eul->taudx.at(id).e22;
                    double taudx23=eul->taudx.at(id).e23;
                    double taudx33=eul->taudx.at(id).e33;
                    
                    double taudy11=eul->taudy.at(id).e11;
                    double taudy12=eul->taudy.at(id).e12;
                    double taudy13=eul->taudy.at(id).e13;
                    double taudy22=eul->taudy.at(id).e22;
                    double taudy23=eul->taudy.at(id).e23;
                    double taudy33=eul->taudy.at(id).e33;
                    
                    double taudz11=eul->taudz.at(id).e11;
                    double taudz12=eul->taudz.at(id).e12;
                    double taudz13=eul->taudz.at(id).e13;
                    double taudz22=eul->taudz.at(id).e22;
                    double taudz23=eul->taudz.at(id).e23;
                    double taudz33=eul->taudz.at(id).e33;
                    
                    /* now looks like calculating a bunch of values for the current particle */

                    ranU=random::norRan();
                    double randXO=pow((CoEps*dt),0.5)*ranU;
                    double randXN=sqrt( (CoEps/(2.0*eigVal_11)) * ( exp(2.0*eigVal_11*dt)- 1.0 ) ) * ranU;
                    
                    ranV=random::norRan();
                    double randYO=pow((CoEps*dt),0.5)*ranV;
                    double randYN=sqrt( (CoEps/(2.0*eigVal_22)) * ( exp(2.0*eigVal_22*dt)- 1.0 ) ) * ranV;
                    
                    ranW=random::norRan();
                    double randZO=pow((CoEps*dt),0.5)*ranW;
                    double randZN=sqrt( (CoEps/(2.0*eigVal_33)) * ( exp(2.0*eigVal_33*dt)- 1.0 ) ) * ranW;                    
                    
                    /* now it calculates the Ax=b stuff, in stages starting at this point! */

                    eul->windP.e11=uPrime;  // I think this is some kind of current velFluct for a current particle?
                    eul->windP.e21=vPrime;
                    eul->windP.e31=wPrime;
                    eul->windPRot=eul->matrixVecMult(eul->eigVecInv.at(id),eul->windP);
                      
                    double URot = eul->windPRot.e11;
                    double VRot = eul->windPRot.e21;
                    double WRot = eul->windPRot.e31;
                    
                    double URot_1st = URot*exp(eigVal_11*dt) - ( (ka0_11/eigVal_11)*( 1.0 - exp(eigVal_11*dt)) ) + randXN;
                    double VRot_1st = VRot*exp(eigVal_22*dt) - ( (ka0_21/eigVal_22)*( 1.0 - exp(eigVal_22*dt)) ) + randYN;
                    double WRot_1st = WRot*exp(eigVal_33*dt) - ( (ka0_31/eigVal_33)*( 1.0 - exp(eigVal_33*dt)) ) + randZN;
                      
                    eul->windPRot.e11 = URot_1st;
                    eul->windPRot.e21 = VRot_1st;
                    eul->windPRot.e31 = WRot_1st;
                    
                    eul->windP=eul->matrixVecMult(eul->eigVec.at(id),eul->windPRot);
                      
                    double U_1st = eul->windP.e11;
                    double V_1st = eul->windP.e21;
                    double W_1st = eul->windP.e31;
                    
                    flag_g2nd=0;
                    if(g2nd_11!=0.0 && U_1st!=0) {
                        if(g2nd_11/fabs(g2nd_11) == U_1st/fabs(U_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(g2nd_21!=0.0 && V_1st!=0) {
                        if(g2nd_21/fabs(g2nd_21) == V_1st/fabs(V_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(g2nd_31!=0.0 && W_1st!=0){
                        if(g2nd_31/fabs(g2nd_31) == W_1st/fabs(W_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(flag_g2nd) { 
                        flag_g2nd=0;
                        double quan1=(1.0-g2nd_11*U_1st*dt);
                        double quan2=(1.0-g2nd_21*V_1st*dt);
                        double quan3=(1.0-g2nd_31*W_1st*dt);
                        
                        if(g2nd_11*U_1st!=0.0 && count<countMax) {
                            if(fabs(quan1)<0.5) {
                                tStepMin=2.0*dt;
                                count++;
                                continue;
                            }
                        }
                        if(g2nd_21*V_1st!=0.0 && count<countMax) {
                            if(fabs(quan2)<0.5) {
                                tStepMin=2.0*dt;
                                count++;
                                continue;
                            }
                        }
                        if(g2nd_31*W_1st!=0.0 && count<countMax) {
                            if(fabs(quan3)<0.5) {
                                tStepMin=dt*2.0;
                                count++;
                                continue;
                            }   
                        }   
                    }
                    double U_2nd = U_1st/(1.0-(g2nd_11*U_1st*dt));
                    double V_2nd = V_1st/(1.0-(g2nd_21*V_1st*dt));
                    double W_2nd = W_1st/(1.0-(g2nd_31*W_1st*dt));
                    
                    double du_3rd=0.5*( lam11*(taudy11*U_2nd*V_2nd + taudz11*U_2nd*W_2nd) 
                                       + lam12*(taudx11*V_2nd*U_2nd + taudy11*V_2nd*V_2nd + taudz11*V_2nd*W_2nd) 
                        	                + lam13*(taudx11*W_2nd*U_2nd + taudy11*W_2nd*V_2nd + taudz11*W_2nd*W_2nd) 
                        	                + lam21*(                      taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd)
                        	                + lam22*(taudx12*V_2nd*U_2nd + taudy12*V_2nd*V_2nd + taudz12*V_2nd*W_2nd) 
                        	                + lam23*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
                        	                + lam31*(                      taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd)
                        	                + lam32*(taudx13*V_2nd*U_2nd + taudy13*V_2nd*V_2nd + taudz13*V_2nd*W_2nd) 
                        	                + lam33*(taudx13*W_2nd*U_2nd + taudy13*W_2nd*V_2nd + taudz13*W_2nd*W_2nd) 
                        	               )*dt;
                    double dv_3rd=0.5*( lam11*(taudx12*U_2nd*U_2nd + taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd) 
                      	                + lam12*(taudx12*V_2nd*U_2nd +                       taudz12*V_2nd*W_2nd) 
                      	                + lam13*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
                      	                + lam21*(taudx22*U_2nd*U_2nd + taudy22*U_2nd*V_2nd + taudz22*U_2nd*W_2nd)
                      	                + lam22*(taudx22*V_2nd*U_2nd +                       taudz22*V_2nd*W_2nd) 
                      	                + lam23*(taudx22*W_2nd*U_2nd + taudy22*W_2nd*V_2nd + taudz22*W_2nd*W_2nd) 
                      	                + lam31*(taudx23*U_2nd*U_2nd + taudy23*U_2nd*V_2nd + taudz23*U_2nd*W_2nd)
                      	                + lam32*(taudx23*V_2nd*U_2nd +                       taudz23*V_2nd*W_2nd) 
                      	                + lam33*(taudx23*W_2nd*U_2nd + taudy23*W_2nd*V_2nd + taudz23*W_2nd*W_2nd) 
                                      )*dt;
                    double dw_3rd=0.5*( lam11*(taudx13*U_2nd*U_2nd + taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd) 
                      	                + lam12*(taudx13*V_2nd*U_2nd + taudy13*V_2nd*V_2nd + taudz13*V_2nd*W_2nd) 
                      	                + lam13*(taudx13*W_2nd*U_2nd + taudy13*W_2nd*V_2nd                      ) 
                      	                + lam21*(taudx23*U_2nd*U_2nd + taudy23*U_2nd*V_2nd + taudz23*U_2nd*W_2nd)
                      	                + lam22*(taudx23*V_2nd*U_2nd + taudy23*V_2nd*V_2nd + taudz23*V_2nd*W_2nd) 
                      	                + lam23*(taudx23*W_2nd*U_2nd + taudy23*W_2nd*V_2nd                      ) 
                      	                + lam31*(taudx33*U_2nd*U_2nd + taudy33*U_2nd*V_2nd + taudz33*U_2nd*W_2nd)
                      	                + lam32*(taudx33*V_2nd*U_2nd + taudy33*V_2nd*V_2nd + taudz33*V_2nd*W_2nd) 
                      	                + lam33*(taudx33*W_2nd*U_2nd + taudy33*W_2nd*V_2nd                      ) 
                                      )*dt;

                    /* I think at this point, it finally does the final calculation on the current velFluct values */

                    uPrime=U_2nd+du_3rd;
                    vPrime=V_2nd+dv_3rd;
                    wPrime=W_2nd+dw_3rd;
                    
                    // now check to see if the value is nan or not. Guess this is the rogue check stuff
                    if(isnan(uPrime) || isnan(vPrime) || isnan(wPrime)) {
                        //std::cerr<<"NAN.....>!!!!!!!"<<std::endl;
                        //std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        //std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        //std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        //std::cerr<<"tStep      : "<<tStep<<std::endl;
                        //std::cerr<<"par        : "<<par<<std::endl;
                        //std::cout<<"eigen Vector "<<std::endl;
                        //eul->display(eul->eigVecInv.at(id));
                        //std::cout<<"eigen Value  "<<std::endl;
                        //eul->display(eul->eigVal.at(id));
                        //exit(1);
                    }
                    
                    // whatever the heck this is, it is hard coded
                    double terFacU = 2.5;
                    double terFacV = 2.;
                    double terFacW = 2.;
                    if(kV>10 && kV<14) {
                        terFacU = 8.5;
                        terFacV = 10.5;
                        terFacW = 10.5;
                    }
                    
                    /* now checking to see if the particle has gone rogue or not */

                    if(fabs(uPrime)>terFacU*fabs(turb->sig.at(id).e11) && countPrm<countPrmMax) {
                        dis->prime.at(par).x = turb->sig.at(id).e11*random::norRan();
                        /*	      std::cout<<"Uprime OLD : "<<uPrime<<std::endl;
                        std::cout<<"Uprime New : "<<dis->prime.at(par).x<<std::endl;
                        std::cout<<"SIGMA U    : "<<turb->sig.at(id).e11<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    if(fabs(vPrime)>terFacV*fabs(turb->sig.at(id).e22)&& countPrm<countPrmMax) {
                        dis->prime.at(par).y=turb->sig.at(id).e22*random::norRan();
                        /*	      std::cout<<"Vprime OLD : "<<vPrime<<std::endl;
                        std::cout<<"Vprime New : "<<dis->prime.at(par).y<<std::endl;
                        std::cout<<"SIGMA V    : "<<turb->sig.at(id).e22<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    if(fabs(wPrime)>terFacW*fabs(turb->sig.at(id).e33)&& countPrm<countPrmMax){
                        dis->prime.at(par).z=turb->sig.at(id).e33*random::norRan();
                        /*	      std::cout<<"Wprime OLD : "<<wPrime<<std::endl;
                        std::cout<<"Wprime New : "<<dis->prime.at(par).z<<std::endl;
                        std::cout<<"SIGMA W    : "<<turb->sig.at(id).e33<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    /* 
                        increment the number of rogue particles for this iteration 
                        guess they weren't keeping track of old information since loopPrm = 0 at each time?
                    */

                    if(flagPrime==1) {
                        flagPrime=0;
                        loopPrm++;
                        continue;
                    }
                    
                    // I think this is finally calculating the new particle positions :) finally something that makes sense again !!!
                    double disX=((uMean+uPrime)*dt);
                    double disY=((vMean+vPrime)*dt);
                    double disZ=((wMean+wPrime)*dt);
                    
                    // now making sure the displacement of the particle isn't bigger than the velocity grid, if so, redo the particle stuff at a halved timestep
                    if(fabs(disX)>urb->grid.dx || fabs(disY)>urb->grid.dy || fabs(disZ)>urb->grid.dz){
                        tStepMin=dt/2.0;
                        loopTby2++;
                        continue;
                    }
                    
                    // if the particle got to this point, it only moved by one wind cell, so now update the particle position information!!!
                    xPos=xPos+disX;
                    yPos=yPos+disY;
                    zPos=zPos+disZ;
                    
                    if(zPos<eul->zo) {
                        Flag=1;
                    }
                    
                    /* this was commented out, but it appears to be the how boundary conditions used to be handled, via straight up reflection
                        importantly, there appears to be a simpler version of reflection, like what I'm used to, and this humongous ugly one that is the 
                        true wall boundary condition! Yet this big and ugly guy is not used apparently, but is sitting there, big and nasty!
                    */
                    //reflection(zPos,wPrime,eul->zo,disX,disY,disZ,xPos,yPos,eul,urb,iV,jV,kV,uPrime,vPrime);
                    
                    // now set variables for the next loop, man this timestep stuff is difficult and confusing to me, 
                    // but this part of setting the old velFluct terms to be the current values from this iteration makes sense
                    // is also setting the current positions there too. Looks like the positions and velFluct change only when ready to be advected
                    // so for the first iteration they haven't been moved yet, they used to be the initial values :)
                    dis->prime.at(par).x=uPrime;
                    dis->prime.at(par).y=vPrime;
                    dis->prime.at(par).z=wPrime;
                    dis->pos.at(par).x=xPos;
                    dis->pos.at(par).y=yPos;
                    dis->pos.at(par).z=zPos;
                    
                    tStepUsed=tStepUsed+dt;
                    tStepRem=tStepRem-dt;
                    dt=tStepRem;
                    tStepMin=tStepInp;
                    loopTby2=0;
                } // if in domain
                else {
                    // if not in the domain, basically the particle is bad/ignored and this is how it is handled
                    tStepRem=0.0;
                    dis->pos.at(par).x=-999.0;
                    dis->pos.at(par).y=-999.0;
                    dis->pos.at(par).z=-999.0;;
                }//if for domain ends   
            } // while (tStepRem>1.0e-5)
        } // for (int par=0; par<parToMove;par++)

        // all particles have been integrated in time by this point

        // this is basically saying, if we are past the time to start averaging values to calculate the concentration,
        // then calculate the average, where average does . . .
        if(timeStepStamp.at(tStep) >= sCBoxTime){
            average(tStep,dis,urb);
        }
        // this is basically saying, if the current time has passed a time that we should be outputting values at,
        // then calculate the concentrations for the sampling boxes, save the output for this timestep, and update time counter for when to do it again.
        // I'm honestly confused why cBox gets set to zero, unless this is meant to cleanup for the next iteration. If this is so, then why do it in a way
        // that you can't output the information if you ever need to debug it? Seems like it should be a temporary variable then.
        if(timeStepStamp.at(tStep)>= sCBoxTime+avgTime) {
            //std::cout<<"loopPrm   :"<<loopPrm<<std::endl;
            //std::cout<<"loopLowestCell :"<<loopLowestCell<<std::endl;
            double cc=(tStepInp)/(avgTime*volume*numPar);
            for(int k=0;k<nBoxesZ;k++) {
                for(int j=0;j<nBoxesY;j++) {
                    for(int i=0;i<nBoxesX;i++) {
                        int id=k*nBoxesY*nBoxesX+j*nBoxesX+i;
                        conc.at(id) = cBox.at(id)*cc;
                        cBox.at(id) = 0.0;
                    }
                }
            }
            save(output);
            avgTime=avgTime+PID->colParams->timeAvg;    // I think this is updating the averaging time for the next loop
        }
    } // for(tStep=0; tStep<numTimeStep; tStep++)
} // run()

#endif  // USE_NEWCODE or no


void Plume::save(Output* output) {
    
    std::cout<<"[Plume] \t Saving particle concentrations"<<std::endl;
    
    // output size and location
    std::vector<size_t> scalar_index;
    std::vector<size_t> scalar_size;
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    scalar_index = {static_cast<unsigned long>(output_counter)};
    scalar_size  = {1};
    vector_index = {static_cast<size_t>(output_counter), 0, 0, 0};
    vector_size  = {1, static_cast<unsigned long>(nBoxesZ),static_cast<unsigned long>(nBoxesY), static_cast<unsigned long>(nBoxesX)};
    
    timeOut = (double)output_counter;
    
    // loop through 1D fields to save
    for (int i=0; i<output_scalar_dbl.size(); i++) {
        output->saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
    }
    
    // loop through 2D double fields to save
    for (int i=0; i<output_vector_dbl.size(); i++) {

        // x,y,z, terrain saved once with no time component
        if (i<3 && output_counter==0) {
            output->saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
        } else {
            output->saveField2D(output_vector_dbl[i].name, vector_index,
                                vector_size, *output_vector_dbl[i].data);
        }
    }

    // remove x, y, z, terrain from output array after first save
    if (output_counter==0) {
        output_vector_dbl.erase(output_vector_dbl.begin(),output_vector_dbl.begin()+3);
    }

    // increment for next time insertion
    output_counter +=1;
}

double Plume::dot(const pos &vecA, const pos &vecB){
    return(vecA.x*vecB.x + vecA.y*vecB.y + vecA.z*vecB.z);
}

pos Plume::normalize(const pos &vec){
    double mag=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
    pos vecTmp;
    vecTmp.x=vec.x/mag;
    vecTmp.y=vec.y/mag;
    vecTmp.z=vec.z/mag;
    return(vecTmp);
}

pos Plume::VecScalarMult(const pos &vec,const double &a){
    pos vecTmp;
    vecTmp.x=a*vec.x;
    vecTmp.y=a*vec.y;
    vecTmp.z=a*vec.z;
    return(vecTmp);
}

pos Plume::posAdd(const pos &vecA,const pos &vecB){
    pos vecTmp;
    vecTmp.x=vecA.x+vecB.x;
    vecTmp.y=vecA.y+vecB.y;
    vecTmp.z=vecA.z+vecB.z;
    return(vecTmp);
}

pos Plume::reflect(const pos &vec,const pos &normal) {
    pos a=VecScalarMult(normal , 2.0*dot(vec, normal));
    a.x=-a.x;
    a.y=-a.y;
    a.z=-a.z;
    pos vecTmp=posAdd(vec,a);
    return(vecTmp);
}

double Plume::distance(const pos &vecA,const pos &vecB) {
    return(sqrt((vecA.x-vecB.x)*(vecA.x-vecB.x) + (vecA.y-vecB.y)*(vecA.y-vecB.y) + (vecA.z-vecB.z)*(vecA.z-vecB.z) ));
}

pos Plume::posSubs(const pos &vecA, const pos & vecB){
    pos vecTmp;
    vecTmp.x=vecA.x-vecB.x;
    vecTmp.y=vecA.y-vecB.y;
    vecTmp.z=vecA.z-vecB.z;
    return(vecTmp);
}

void Plume::average(const int tStep,const Dispersion* dis, const Urb* urb) {
    for(int i=0;i<numPar;i++) {
        if(tStrt.at(i)>timeStepStamp.at(tStep)) continue;
        double xPos=dis->pos.at(i).x;
        double yPos=dis->pos.at(i).y;
        double zPos=dis->pos.at(i).z;
        if(zPos==-1) continue;
        int iV=int(xPos/urb->grid.dx);
        int jV=int(yPos/urb->grid.dy);
        int kV=int(zPos/urb->grid.dz)+1;
        int idx=(int)((xPos-lBndx)/boxSizeX);
        int idy=(int)((yPos-lBndy)/boxSizeY);
        int idz=(int)((zPos-lBndz)/boxSizeZ);
        if(xPos<lBndx) idx=-1;
        if(yPos<lBndy) idy=-1;
        if(zPos<lBndz) idz=-1;
        int id=0;
        if(idx>=0 && idx<nBoxesX && idy>=0 && idy<nBoxesY && idz>=0 && idz<nBoxesZ && tStrt.at(i)<=timeStepStamp.at(tStep)) {
            id=idz*nBoxesY*nBoxesX+idy*nBoxesX+idx;
            cBox.at(id)=cBox.at(id)+1.0;
        }
    }
}

double Plume::min(double arr[],int len) {
    double min=arr[0];
    for(int i=1;i<len;i++) {
        if(arr[i]<min) {
            min=arr[i];
        }
    }
    return min;
}

double Plume::max(double arr[],int len) {
    double max=arr[0];
    for(int i=1;i<len;i++) {
        if(arr[i]>max) {
            max=arr[i];
        }
    }
    return max;
}

void Plume::reflection(double &zPos, double &wPrime, const double &z0,const double &disX
		,const double &disY,const double &disZ ,double &xPos
		,double &yPos,const Eulerian* eul, const Urb* urb, const int &imc, const int &jmc
		, const int &kmc,double &uPrime,double &vPrime){
    
    //shader reflection   
    //Now do Reflection		
    //	pos u;
    //point of intersection
    //	pos pI;	
    //incident vector
    //	pos l;
    //reflection vector
    //	pos r;
    //normal vector
    //	pos normal;
    //distance from reflected surface
    //	float dis;
    
    //	float denom;
    //	float numer;
    
    //	vec2 cIndex;


    // have to reflect all these values, not just the positions and the velocities. That is why this function looks so dang complicated!

    
    pos u,n,vecS,prevPos,normal,vecZ,vecTmp,vecTmp1,pI,r,l,pos,prmCurr;
    double d,denom,numer,dis;
    pos.x=xPos;
    pos.y=yPos;
    pos.z=zPos;
            
    prevPos.x=xPos-disX;
    prevPos.y=yPos-disY;
    prevPos.z=zPos-disZ;

    prmCurr.x=uPrime;
    prmCurr.y=vPrime;
    prmCurr.z=wPrime;
    
    int i = int(xPos/urb->grid.dx);
    int j = int(yPos/urb->grid.dy);
    int k = int(zPos/urb->grid.dz)+1;
    int id=0;
    
    double eps_S = 0.0001;
    double eps_d = 0.01;
    double smallestS = 100.0;
    double cellBld = 1.0;
    
    //check if within domain
    if((i < nx) && (j < ny) && (k < nz) && (i >= 0) && (j >= 0)) {
        double cellBld = 1.0; //set it so that default is no reflection
        
        //ground is at 0 in this code, this also avoids out of bound in case of large negative k
        if(k < 0) k = 0;
        
        id=k*nx*ny+j*nx+i;

        if(k >= 0) {
            //Perform lookup into wind texture to see if new position is inside a building
            cellBld = urb->grid.icell.at(id);
        }
        int count=0;
        
        //pos.z<0.0 covers ground reflections
        while((urb->grid.icell.at(id)==0 || (zPos < 0.0)) && count<25) {
        
            /*	  std::cout<<"Before Reflection-prev"<<xPos-disX<<"   "<<yPos-disY<<"   "<<zPos-disZ<<std::endl;
            std::cout<<std::endl;
            std::cout<<"Before Reflection"<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
            std::cout<<"Before Reflection"<<uPrime<<"   "<<vPrime<<"   "<<wPrime<<std::endl;*/
            
            //	  std::cout<<"reflection while, count:"<<count<<std::endl;
            count=count+1;
            
            // pos(pos) - prevPos;// u has disX,disY and disZ
            u.x =disX;
            u.y =disY;
            u.z =disZ;
                
            double s1 = -1.0; //for -x
            double s2 = -1.0; //for +x
            double s3 = -1.0; //for -y
            double s4 = -1.0; //for +y
            double s5 = -1.0; //for +z (buildings)
            double s6 = -1.0; //Not used
            double s7 = -1.0; //for ground
                 
	        smallestS = 100.0;
	        double xfo=-999.0;
	        double yfo=-999.0;
	        double zfo=-999.0;
	        double ht =-999.0;
	        double wti=-999.0;
	        double lti=-999.0;
            
            //	  std::cout<<"reflection while, build params,cellBld:"<<cellBld<<"  "<<id<<"  "<<i<<"  "<<j<<"  "<<k<<std::endl;
            //if(cellBld!=-1) {
	        //    xfo = utl.xfo.at(int(cellBld));//bcoords.x;
	        //    yfo = utl.yfo.at(int(cellBld));//bcoords.y;
	        //    zfo = utl.zfo.at(int(cellBld));//bcoords.z;
	        //    ht  = utl.hgt.at(int(cellBld));//bdim.x;
	        //    wti = utl.wth.at(int(cellBld));//bdim.y;
	        //    lti = utl.len.at(int(cellBld));//bdim.z;
            //}
            //	  std::cout<<"xfo:"<<"  "<<xfo<<" yfo:"<<"  "<<yfo<<" zfo:"<<"  "<<zfo<<std::endl;
                
            //-x normal  
            n.x=-1.0;
            n.y=0.0;
            n.z=0.0;
            vecS.x=xfo;
            vecS.y=0.0;
            vecS.z=0.0;
            
            d = -dot(n,vecS);
            denom = dot(n,u);
            numer = dot(n,prevPos) + d;
            s1 = -numer/denom;
            
            /*	  std::cout<<"-x:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            std::cout<<"-x:s1:"<<"  "<<s1<<std::endl;*/
                  
            //+x normal
            n.x=1.0;
            n.y=0.0;
            n.z=0.0;
            vecS.x=xfo+lti;
            vecS.y=0.0;
            vecS.z=0.0;
            d = -dot(n,vecS);
            denom = dot(n,u);
            numer = dot(n,prevPos) + d;
            s2 = -numer/denom;
            
            /*std::cout<<"+x:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            std::cout<<"s2:"<<"  "<<s2<<std::endl;*/
            
            //+y normal
            n.x=0.0;
            n.y=1.0;
            n.z=0.0;
            
            vecS.x=xfo;
            vecS.y=yfo+(wti/2.0);
            vecS.z=0.0;
                
            d = -dot(n,vecS);
            denom = dot(n,u);
            numer = dot(n,prevPos) + d;
            s3 = -numer/denom;
            
            /*std::cout<<"+y:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            std::cout<<"s3:"<<"  "<<s3<<std::endl;*/
                
            //-y normal
            n.x=0.0;
            n.y=-1.0;
            n.z=0.0;
            
            vecS.x=xfo;
            vecS.y=yfo-(wti/2.0);
            vecS.z=0.0;
            d = -dot(n,vecS);
            denom = dot(n,u);
            numer = dot(n,prevPos) + d;
            s4 = -numer/denom;
            
            /*std::cout<<"-y:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            std::cout<<"s4:"<<"  "<<s4<<std::endl;*/
            
            //+z normal
            n.x=0.0;
            n.y=0.0;
            n.z=1.0;
            
            vecS.x=xfo;
            vecS.y=0.0;
            vecS.z=zfo+ht;
                
            d = -dot(n,vecS);
            denom = dot(n,u);
            numer = dot(n,prevPos) + d;
            s5 = -numer/denom;
            
            //std::cout<<"+z:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            //std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            //std::cout<<"s5:"<<"  "<<s5<<std::endl;
            
            //Ground plane
            n.x=0.0;
            n.y=0.0;
            n.z=1.0;
            numer = dot(n,prevPos);
            denom = dot(n,u);
            s7 = -numer/denom;
            
            //std::cout<<"gr:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
            //std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
            //std::cout<<"s7:"<<"  "<<s7<<std::endl;   
            
            if((s1 < smallestS) && (s1 >= -eps_S)){
              smallestS = s1;
              normal.x=-1.0;
              normal.y=0.0;
              normal.z=0.0;
              //std::cout<<"s1-smallestS:"<<"  "<<smallestS<<std::endl;
            }
            if((s2 < smallestS) && (s2 >= -eps_S)){
              normal.x =1.0;
              normal.y=0.0;
              normal.z=0.0;
              smallestS = s2;
              //std::cout<<"s2-smallestS:"<<"  "<<smallestS<<std::endl;
            }
            if((s3 < smallestS) && (s3 >= -eps_S)){
              normal.x=0.0;
              normal.y=1.0;
              normal.z=0.0;
              smallestS = s3;
              //std::cout<<"s3-smallestS:"<<"  "<<smallestS<<std::endl;
            }	
            if((s4 < smallestS) && (s4 >= -eps_S)){
              normal.x=0.0;
              normal.y=-1.0;
              normal.z=0.0;
              smallestS = s4;
              //std::cout<<"s4-smallestS:"<<"  "<<smallestS<<std::endl;
            }	   
            if((s5 < smallestS) && (s5 >= -eps_S)){
              normal.x =0.0;
              normal.y=0.0;
              normal.z=1.0;
              smallestS = s5;
              //std::cout<<"s5-smallestS:"<<"  "<<smallestS<<std::endl;
            }	 
            //std::cout<<"normal:"<<"  "<<normal.x<<"   "<<normal.y<<"  "<<normal.z<<std::endl;
            //std::cout<<"smallestS:"<<"  "<<smallestS<<std::endl;
            
            //Detect Edge Collision
            
            double edgeS = fabs(smallestS-s7);
            //std::cout<<"edgeS:"<<"  "<<edgeS<<", eps_d:"<<eps_d<<std::endl;
            if((edgeS < eps_d)){
              //smallestS = s6;
              vecZ.x=0.0;
              vecZ.y=0.0;
              vecZ.z=1.0;
              vecTmp.x=normal.x+vecZ.x;
              vecTmp.y=normal.y+vecZ.y;
              vecTmp.z=normal.z+vecZ.z;
              //std::cout<<"first cond"<<std::endl;
              
              normal = normalize(vecTmp);
            }
            else if((s7 < smallestS) && (s7 >= -eps_S)){
              //std::cout<<"else cond"<<std::endl;
              normal.x=0.0;
              normal.y=0.0;
              normal.z=1.0;
              smallestS = s7;
            }
            //std::cout<<"edge and else:normal:"<<"  "<<normal.x<<"   "<<normal.y<<"  "<<normal.z<<std::endl;
            //std::cout<<"after edge: smallestS:"<<"  "<<smallestS<<std::endl;
            vecTmp1 = VecScalarMult(u,smallestS);
            pI=posAdd(vecTmp1,prevPos);
            //std::cout<<"pI:"<<"  "<<pI.x<<"   "<<pI.y<<"  "<<pI.z<<std::endl;
                
            if((smallestS >= -eps_S) && (smallestS <= eps_S)){
              pI = prevPos;
              r = normal;
            }	
            else{
              l = normalize(posSubs(pI,prevPos));
              r = normalize(reflect(l,normal));
            }
            //std::cout<<"l:"<<"  "<<l.x<<"   "<<l.y<<"  "<<l.z<<std::endl;
            //std::cout<<"r:"<<"  "<<r.x<<"   "<<r.y<<"  "<<r.z<<std::endl;
            
            dis = distance(pI,pos);		
            //std::cout<<"dis:"<<"  "<<dis<<std::endl;
            
            prevPos = pI;
            pos = posAdd(pI,VecScalarMult(r,dis));
            //update xpos,ypos,zpos
            xPos=pos.x;
            yPos=pos.y;
            zPos=pos.z;
            prmCurr = reflect(prmCurr,normal);
            uPrime=prmCurr.x;
            vPrime=prmCurr.y;
            wPrime=prmCurr.z;
            //update primes
            i = int(pos.x);
            j = int(pos.y);
            k = int(pos.z)+1;
        
            //NOTE: Consider what happens if building is too close to domain.
            //Do check to make sure i,j,k's are valid;
            cellBld = 1.0;
            if(k < 0) k = 0;
            
            id=k*nx*ny+j*nx+i;
            if(k >= 0) {
              //std::cout<<"end of while cellbld check"<<i<<"  "<<j<<"   "<<k<<std::endl;
              cellBld = urb->grid.icell.at(id);  //find cellType of (i,j,k) {cellType stores
            }
            //std::cout<<"after Reflection"<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
            //std::cout<<"after Reflection"<<uPrime<<"   "<<vPrime<<"   "<<wPrime<<std::endl;
                //          std::cout<<"count:"<<count<<std::endl;
            if(smallestS>=99.999 || count>20) {
                //std::cout<<"may be a reflection problem"<<std::endl;
                //std::cout<<"count:"<<count<<std::endl;
                //std::cout<<"smallestS:"<<smallestS<<std::endl;
            }
        } //while loop for reflection
    } //domain check
    
    return;
}
