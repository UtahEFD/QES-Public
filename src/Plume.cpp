//
//  Plume.cpp
//  
//  This class handles plume model
//

#include "Plume.hpp"

Plume::Plume(Urb* urb,Dispersion* dis, PlumeInputData* PID, Output* output) {
    
    std::cout<<"[Plume] \t Setting up particles "<<std::endl;
    
    // make local copies
    nx = urb->grid.nx;
    ny = urb->grid.ny;
    nz = urb->grid.nz;

    // get the urb domain start and end values, needed for wall boundary condition application
    domainXstart = urb->domainXstart;
    domainXend = urb->domainXend;
    domainYstart = urb->domainYstart;
    domainYend = urb->domainYend;
    domainZstart = urb->domainZstart;
    domainZend = urb->domainZend;
    
    /* setup the sampling box concentration information */

    sCBoxTime = PID->colParams->timeStart;
    avgTime  = PID->colParams->timeAvg;

    nBoxesX = PID->colParams->nBoxesX;
    nBoxesY = PID->colParams->nBoxesY;
    nBoxesZ = PID->colParams->nBoxesZ;
    
    boxSizeX = (PID->colParams->boxBoundsX2-PID->colParams->boxBoundsX1)/nBoxesX;	  
    boxSizeY = (PID->colParams->boxBoundsY2-PID->colParams->boxBoundsY1)/nBoxesY;	  
    boxSizeZ = (PID->colParams->boxBoundsZ2-PID->colParams->boxBoundsZ1)/nBoxesZ;
    
    volume = boxSizeX*boxSizeY*boxSizeZ;
    
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
    
    
    int zR = 0;
    int yR = 0;
    int xR = 0;
    for(int k = 0; k < nBoxesZ; ++k)
    {
        zBoxCen.at(k) = lBndz + (zR*quanZ) + (boxSizeZ/2.0);        // the .at(k) is different from [k] in that it checks the bounds, throwing an out of range error if outside the bounds of the vector
        zR++;
    }
    for(int j = 0; j < nBoxesY; ++j)
    {
        yBoxCen.at(j) = lBndy + (yR*quanY) + (boxSizeY/2.0);
        yR++;
    }
    for(int i = 0; i <nBoxesX; ++i)
    {
        xBoxCen.at(i) = lBndx + (xR*quanX) + (boxSizeX/2.0);
        xR++;
    }

    cBox.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    conc.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    
    
    /* make copies of important input and dispersion information for particle release */

    dt = PID->simParams->timeStep;
    numTimeStep  = dis->numTimeStep;

#if 0
    numPar = dis->numPar;    
    
    tStrt.resize(numPar);
    for(int i = 0; i < numPar; i++)
    {
        tStrt.at(i) = dis->pointList.at(i).tStrt;
    }
#endif
    
    timeStepStamp.resize(numTimeStep);
    timeStepStamp = dis->timeStepStamp;
    

    parPerTimestep.resize(numTimeStep);
    parPerTimestep = dis->parPerTimestep;
    

    
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
    AttVectorDbl att_conc  = {&conc,    "conc", "concentration","--", dim_vector};

#if 0
    AttVectorDbl att_px     = {&xBoxCen, "xp", "x-position of particle","m",dim_scalar_x};
    AttVectorDbl att_py     = {&yBoxCen, "yp", "y-position of particle","m",dim_scalar_y};
    AttVectorDbl att_pz     = {&zBoxCen, "zp", "z-position of particle","m",dim_scalar_z};
#endif
    
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


    // now get the input boundary condition types from the inputs
    std::string xBCtype = PID->BCs->xBCtype;
    std::string yBCtype = PID->BCs->yBCtype;
    std::string zBCtype = PID->BCs->zBCtype;

    // now set the boundary condition function for the plume runs, checking to make sure the input BCtypes are legitimate
    setBCfunctions(xBCtype,yBCtype,zBCtype);


}


void Plume::run(Urb* urb, Turb* turb, Eulerian* eul, Dispersion* dis, PlumeInputData* PID, Output* output)
{
    std::cout << "[Plume] \t Advecting particles " << std::endl;

    // get the threshold velocity fluctuation to define rogue particles from dispersion class
    double vel_threshold = dis->vel_threshold;
    
    // //////////////////////////////////////////
    // TIME Stepping Loop
    // for every time step
    // //////////////////////////////////////////

    // I want to get an idea of the overall time of the time integration loop
    auto timerStart_timeIntegration = std::chrono::high_resolution_clock::now();
        
    for(int tStep = 0; tStep < numTimeStep; tStep++)
    {
        // 
        // Add new particles now
        // - walk over all sources and add the emitted particles from
        // each source to the overall particle list
        // 
        auto timerStart_particleRelease = std::chrono::high_resolution_clock::now();
        std::vector<particle> nextSetOfParticles(0);
        for (auto sidx=0u; sidx < dis->allSources.size(); sidx++) {
            int numParticles = dis->allSources[sidx]->emitParticles( (float)dt, (float)(tStep*dt), nextSetOfParticles );
            if (numParticles > 0)
                std::cout << "Emitting " << numParticles << " particles from source " << sidx << std::endl;
        }
        
        dis->setParticleVals( turb, eul, nextSetOfParticles );
        
        // append all the new particles on to the big particle
        // advection list
        dis->pointList.insert( dis->pointList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end() );

        auto timerEnd_particleRelease = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_particleRelease = timerEnd_particleRelease - timerStart_particleRelease;
        std::cout << "finished emitting particles from sources. Total numParticles = \"" << dis->pointList.size() << "\"" << std::endl;
        std::cout << "\telapsed time: " << elapsed_particleRelease.count() << " s" << std::endl;   // Print out elapsed execution time

        // Move each particle for every time step
        double isRogueCount = dis->isRogueCount;    // This probably could be moved from dispersion to one level back in this for loop
        double isActiveCount = dis->isActiveCount;

        // Advection Loop

        // I want to get an idea of the overall time of the advection loop, and different parts of the advection loop
        auto timerStart_advection = std::chrono::high_resolution_clock::now();

        for (int par=0; par<dis->pointList.size(); par++)
        {
            // get the current isRogue and isActive information
            // in this whole section, the idea of having single value temporary storage instead of just referencing values
            //  directly from the dispersion class seems a bit strange, but it makes the code easier to read cause smaller variable names
            //  also, in theory it is faster?
            bool isRogue = dis->pointList.at(par).isRogue;
            bool isActive = dis->pointList.at(par).isActive;

            // first check to see if the particle should even be advected
            if(isActive == true && isRogue == false)
            {

                // overall particle timer
                auto timerStart_particle = std::chrono::high_resolution_clock::now();


                // this is getting the current position for where the particle is at for a given time
                // if it is the first time a particle is ever released, then the value is already set at the initial value
                double xPos = dis->pointList.at(par).pos.e11;
                double yPos = dis->pointList.at(par).pos.e21;
                double zPos = dis->pointList.at(par).pos.e31;

                // this is the old velFluct value
                // hmm, Brian's code just starts out setting these values to zero,
                // so the prime values are actually the old velFluct. velFluct_old and prime are probably identical and kind of redundant in this implementation
                // but it shouldn't hurt anything for now, even if it is redundant
                // besides, it will probably change a bit once I figure out what exactly I want outputted on a regular, and on a debug basis
                double uPrime = dis->pointList.at(par).prime.e11;
                double vPrime = dis->pointList.at(par).prime.e21;
                double wPrime = dis->pointList.at(par).prime.e31;

                // should also probably grab and store the old values in this same way
                // these consist of velFluct_old and tao_old
                // also need to keep track of a delta_velFluct and an isActive flag for each particle
                // though delta_velFluct doesn't need grabbed as a value till later now that I think on it
                double uFluct_old = dis->pointList.at(par).prime_old.e11;
                double vFluct_old = dis->pointList.at(par).prime_old.e21;
                double wFluct_old = dis->pointList.at(par).prime_old.e31;
                double txx_old = dis->pointList.at(par).tau_old.e11;
                double txy_old = dis->pointList.at(par).tau_old.e12;
                double txz_old = dis->pointList.at(par).tau_old.e13;
                double tyy_old = dis->pointList.at(par).tau_old.e22;
                double tyz_old = dis->pointList.at(par).tau_old.e23;
                double tzz_old = dis->pointList.at(par).tau_old.e33;
                
                
                /*
                    now get the values for the current iteration
                    will need to use the interp3D function
                    Need to get velMean, CoEps, tao, flux_div
                    Then need to call makeRealizable on tao then calculate inverse tao
                */


                // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
                // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
                eul->setInterp3Dindexing(dis->pointList.at(par).pos);



                // this is the Co times Eps for the particle
                double CoEps = eul->interp3D(turb->CoEps,"CoEps");
                //double CoEps = eul->interp3D(turb->CoEps,"Eps");
                
                
                // this is the current velMean value
                Wind velMean = eul->interp3D(urb->wind);
                double uMean = velMean.u;
                double vMean = velMean.v;
                double wMean = velMean.w;
                
                // this is the current reynolds stress tensor
                matrix6 tao = eul->interp3D(turb->tau);
                double txx = tao.e11;
                double txy = tao.e12;
                double txz = tao.e13;
                double tyy = tao.e22;
                double tyz = tao.e23;
                double tzz = tao.e33;
                
                // now need flux_div_vel, not the different dtxxdx type components
                vec3 flux_div = eul->interp3D(eul->flux_div);
                double flux_div_x = flux_div.e11;
                double flux_div_y = flux_div.e21;
                double flux_div_z = flux_div.e31;


                // now need to call makeRealizable on tao
                // note that the invarianceTol is hard coded for now, but needs to be added as an overall input to CUDA-Plume
                double invarianceTol = 1e-10;
                matrix6 realizedTao = makeRealizable(tao,invarianceTol);
                txx = realizedTao.e11;
                txy = realizedTao.e12;
                txz = realizedTao.e13;
                tyy = realizedTao.e22;
                tyz = realizedTao.e23;
                tzz = realizedTao.e33;

                // now need to calculate the inverse values for tao
                // I'm probably going to write my own function, but it looks like using the structs might be the best way to control output
                // either that or passing in the variables by reference that need changed
                matrix9 fullTao;
                fullTao.e11 = txx;
                fullTao.e12 = txy;
                fullTao.e13 = txz;
                //fullTao.e21 = txy;
                fullTao.e22 = tyy;
                fullTao.e23 = tyz;
                //fullTao.e31 = txz;
                //fullTao.e32 = tyz;
                fullTao.e33 = tzz;

                fullTao.e21 = 0.0;
                fullTao.e31 = 0.0;
                fullTao.e32 = 0.0;

                // I just noticed that Brian's code always leaves the last three components alone, never filled with new tensor info
                // even though he keeps everything as a matrix9 datatype. This seems fine for makeRealizable, but I wonder if it
                // messes with the invert3 stuff since those values are used even though they are empty
                // going to send in the matrix with all 9 terms anyways, and use the method that assumes they are all filled,
                // so like Brian's code in method, not the same with the inputs to the function.
                matrix9 inverseTao = invert3(fullTao);
                double lxx = inverseTao.e11;
                double lxy = inverseTao.e12;
                double lxz = inverseTao.e13;
                double lyy = inverseTao.e22;
                double lyz = inverseTao.e23;
                double lzz = inverseTao.e33;




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
                matrix9 A;
                vec3 b;

                A.e11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*dt;
                A.e12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*dt;
                A.e13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*dt;

                A.e21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*dt;
                A.e22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*dt;
                A.e23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*dt;
                
                A.e31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*dt;
                A.e32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*dt;
                A.e33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*dt;


                b.e11 = -uFluct_old - 0.50*flux_div_x*dt - sqrt(CoEps*dt)*xRandn;
                b.e21 = -vFluct_old - 0.50*flux_div_y*dt - sqrt(CoEps*dt)*yRandn;
                b.e31 = -wFluct_old - 0.50*flux_div_z*dt - sqrt(CoEps*dt)*zRandn;


                // now prepare for the Ax=b calculation by calculating the inverted A matrix
                matrix9 Ainv = invert3(A);


                // now do the Ax=b calculation using the inverted matrix
                // hm, since getting out three values, might be easier to do a pass in by reference thing for the velPrime values
                vec3 velPrime = matmult(Ainv,b);
                uPrime = velPrime.e11;
                vPrime = velPrime.e21;
                wPrime = velPrime.e31;
                

                // now check to see if the value is rogue or not
                if( abs(uPrime) >= vel_threshold || isnan(uPrime) && nx > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible uFluct was \"" << uPrime << "\"" << std::endl;
                    uPrime = 0.0;
                    isRogue = true;
                }
                if( abs(vPrime) >= vel_threshold || isnan(vPrime) && ny > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible vFluct was \"" << vPrime << "\"" << std::endl;
                    vPrime = 0.0;
                    isRogue = true;
                }
                if( abs(wPrime) >= vel_threshold || isnan(wPrime) && nz > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible wFluct was \"" << wPrime << "\"" << std::endl;
                    wPrime = 0.0;
                    isRogue = true;
                }

                // Do you need this???
                // ONLY if this should never happen....
                //    assert( isRogue == false );
                
                // now update the particle position for this iteration
                // at some point in time, need to do a CFL condition for only moving one eulerian grid cell at a time
                // as well as a separate CFL condition for the particle timestep
                // this would mean adding some kind of while loop, with an adaptive timestep, controlling the end of the while loop
                //  with the sampling time increment. This means all particles can do multiple time iterations, each with their own timestep
                // but at the sampling timestep, particles are allowed to catch up so they are all calculated by that time
                // currently, we use the sampling timestep so we may violate the eulerian grid CFL condition. But is simpler to work with when getting started
                double disX = ((uMean + uPrime)*dt);
                double disY = ((vMean + vPrime)*dt);
                double disZ = ((wMean + wPrime)*dt);
                
                xPos = xPos + disX;
                yPos = yPos + disY;
                zPos = zPos + disZ;


                // I want to get an idea of the overall time of the advection loop, and different parts of the advection loop
                //std::cout << "applying wallBC" << std::endl;
                //auto timerStart_wallBC = std::chrono::high_resolution_clock::now();

                // now apply boundary conditions
                // notice that this is the old fashioned style for calling a pointer function
                (this->*enforceWallBCs_x)(xPos,uPrime,uFluct_old,isActive, domainXstart,domainXend);
                (this->*enforceWallBCs_y)(yPos,vPrime,vFluct_old,isActive, domainYstart,domainYend);
                (this->*enforceWallBCs_z)(zPos,wPrime,wFluct_old,isActive, domainZstart,domainZend);
                
                // now set the particle values for if they are rogue or outside the domain
                setFinishedParticleVals(xPos,yPos,zPos, isActive,isRogue);

                
                //auto timerEnd_wallBC = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> elapsed_wallBC = timerEnd_wallBC - timerStart_wallBC;
                //std::cout << "wallBC for particle par[" << par << "] and timeStepStamp[" << timeStepStamp.at(tStep) << "] finished" << std::endl;
                //std::cout << "\telapsed time: " << elapsed_wallBC.count() << " s" << std::endl;   // Print out elapsed execution time



                // now update the old values and current values in the dispersion storage to be ready for the next iteration
                // also calculate the velFluct increment
                dis->pointList.at(par).prime.e11 = uPrime;
                dis->pointList.at(par).prime.e21 = vPrime;
                dis->pointList.at(par).prime.e31 = wPrime;
                dis->pointList.at(par).pos.e11 = xPos;
                dis->pointList.at(par).pos.e21 = yPos;
                dis->pointList.at(par).pos.e31 = zPos;

                dis->pointList.at(par).delta_prime.e11 = uPrime - uFluct_old;
                dis->pointList.at(par).delta_prime.e21 = vPrime - vFluct_old;
                dis->pointList.at(par).delta_prime.e31 = wPrime - wFluct_old;
                dis->pointList.at(par).prime_old.e11 = uPrime;
                dis->pointList.at(par).prime_old.e21 = vPrime;
                dis->pointList.at(par).prime_old.e31 = wPrime;

                dis->pointList.at(par).tau_old.e11 = txx;
                dis->pointList.at(par).tau_old.e12 = txy;
                dis->pointList.at(par).tau_old.e13 = txz;
                dis->pointList.at(par).tau_old.e22 = tyy;
                dis->pointList.at(par).tau_old.e23 = tyz;
                dis->pointList.at(par).tau_old.e33 = tzz;

                // now update the isRogueCount
                if(isRogue == true)
                {
                    isRogueCount = isRogueCount + 1;
                }
                if(isActive == false)
                {
                    isActiveCount = isActiveCount + 1;
                }


                dis->pointList.at(par).isRogue = isRogue;
                dis->pointList.at(par).isActive = isActive;


                auto timerEnd_particle = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_particle = timerEnd_particle - timerStart_particle;
                std::cout << "particle iteration par[" << par << "] finished. timestep = \"" << timeStepStamp.at(tStep) << "\"" << std::endl;
                std::cout << "\telapsed time: " << elapsed_particle.count() << " s" << std::endl;   // Print out elapsed execution time
            
            }   // if isActive == true and isRogue == false


        } // for (int par=0; par<parToMove;par++)

        // set the isRogueCount for the time iteration in the disperion data
        // hm, I'm almost wondering if this needs to go into dispersion, could just be kept locally,
        // but declared outside the loop to preserve the value. Depends on the requirements for output and debugging
        dis->isRogueCount = isRogueCount;
        dis->isActiveCount = isActiveCount;


        auto timerEnd_advection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_advection = timerEnd_advection - timerStart_advection;
        std::cout << "advection loop for timeStepStamp[" << timeStepStamp.at(tStep) << "] finished" << std::endl;
        std::cout << "\telapsed time: " << elapsed_advection.count() << " s" << std::endl;   // Print out elapsed execution time


        
        // this is basically saying, if we are past the time to start averaging values to calculate the concentration,
        // then calculate the average, where average does . . .
        if( timeStepStamp.at(tStep) >= sCBoxTime )
        {
            average(tStep,dis,urb);
        }

        // this is basically saying, if the current time has passed a time that we should be outputting values at,
        // then calculate the concentrations for the sampling boxes, save the output for this timestep, and update time counter for when to do it again.
        // I'm honestly confused why cBox gets set to zero, unless this is meant to cleanup for the next iteration. If this is so, then why do it in a way
        // that you can't output the information if you ever need to debug it? Seems like it should be a temporary variable then.
        if(timeStepStamp.at(tStep) >= sCBoxTime+avgTime )
        {
            //std::cout<<"loopPrm   :"<<loopPrm<<std::endl;
            //std::cout<<"loopLowestCell :"<<loopLowestCell<<std::endl;
            double cc = (dt)/(avgTime*volume* dis->pointList.size() );
            for(int k = 0; k < nBoxesZ; k++)
            {
                for(int j = 0; j < nBoxesY; j++)
                {
                    for(int i = 0; i < nBoxesX; i++)
                    {
                        int id = k*nBoxesY*nBoxesX + j*nBoxesX + i;
                        conc.at(id) = cBox.at(id)*cc;
                        cBox.at(id) = 0.0;
                    }
                }
            }
            save(output);
            avgTime = avgTime + PID->colParams->timeAvg;    // I think this is updating the averaging time for the next loop
        }

        int updateFrequency = 1;
        if( tStep % updateFrequency == 0 || tStep == numTimeStep - 1 )
        {
            std::cout << "time = \"" << timeStepStamp.at(tStep) << "\", isRogueCount = \"" << dis->isRogueCount << "\", isActiveCount = \"" << dis->isActiveCount << "\"" << std::endl;
        }


        // 
        // For all particles that need to be removed from the particle
        // advection, remove them now
        //
        // Purge the advection list of all the unneccessary particles....

    } // for(tStep=0; tStep<numTimeStep; tStep++)

    auto timerEnd_timeIntegration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_timeIntegration = timerEnd_timeIntegration - timerStart_timeIntegration;
    std::cout << "time integration loop finished" << std::endl;
    std::cout << "\telapsed time: " << elapsed_timeIntegration.count() << " s" << std::endl;   // Print out elapsed execution time

    // if the debug output folder is an empty string "", the debug output variables won't be written
    writeSimInfoFile(dis,timeStepStamp.at(numTimeStep-1));
    dis->outputVarInfo_text();

}

vec3 Plume::calcInvariants(const matrix6& tau)
{

    vec3 invariants;

    invariants.e11 = tau.e11 + tau.e22 + tau.e33;

    invariants.e21 = tau.e11*tau.e22 + tau.e11*tau.e33 + tau.e22*tau.e33 - tau.e12*tau.e12 - tau.e13*tau.e13 - tau.e23*tau.e23;

    invariants.e31 = tau.e11*(tau.e22*tau.e33 - tau.e23*tau.e23) - tau.e12*(tau.e12*tau.e33 - tau.e23*tau.e13) + tau.e13*(tau.e12*tau.e23 - tau.e22*tau.e13);

    return invariants;
}

matrix6 Plume::makeRealizable(const matrix6& tau,const double& invarianceTol)
{
    // first calculate the invariants and see if they are already realizable

    vec3 invariants = calcInvariants(tau);
    if( invariants.e11 > invarianceTol && invariants.e21 > invarianceTol && invariants.e31 > invarianceTol )
    {
        std::cout << "tau already realizable" << std::endl;
        return tau;     // tau is already realizable
    }

    // since tau is not already realizable, need to make it realizeable
    // start by making a guess of ks, the subfilter scale tke
    // I keep wondering if we can use the input Turb->tke for this or if we should leave it as is
    double b = 4.0/3.0*(tau.e11 + tau.e22 + tau.e33);   // I think this is 4.0/3.0*invariants.e11 as well
    double c = tau.e11*tau.e22 + tau.e11*tau.e33 + tau.e22*tau.e33 - tau.e12*tau.e12 - tau.e13*tau.e13 - tau.e23*tau.e23;   // this is probably invariants.e21
    double ks = 1.01*(-b + sqrt(b*b - 16.0/3.0*c)) / (8.0/3.0);

    // if the initial guess is bad, use the straight up invariance.e11 value
    if( ks < invarianceTol || isnan(ks) )
    {
        ks = 0.5*abs(tau.e11 + tau.e22 + tau.e33);  // looks like 0.5*abs(invariants.e11)
    }

    // now set the initial values of the new make realizable tau, 
    // the first iteration of the following loop
    // notice that through all this process, only the diagonals are really increased by a value of 0.05% of the subfilter tke ks
    matrix6 tau_new;
    tau_new.e11 = tau.e11 + 2.0/3.0*ks;
    tau_new.e22 = tau.e22 + 2.0/3.0*ks;
    tau_new.e33 = tau.e33 + 2.0/3.0*ks;
    tau_new.e12 = tau.e12;
    tau_new.e13 = tau.e13;
    tau_new.e23 = tau.e23;

    // adjust the invariants for the new tau
    invariants = calcInvariants(tau_new);

    // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till tau is realizable
    // or if too many iterations go on, give a warning. I've had trouble with this taking too long
    // if it isn't realizable, so maybe another approach for when the iterations are reached might be smart
    int iter = 0;
    while( (invariants.e11 < invarianceTol || invariants.e21 < invarianceTol || invariants.e31 < invarianceTol) && iter < 1000 )
    {
        iter = iter + 1;

        ks = ks*1.050;      // increase subfilter tke by 5%

        tau_new.e11 = tau.e11 + 2.0/3.0*ks;
        tau_new.e22 = tau.e22 + 2.0/3.0*ks;
        tau_new.e33 = tau.e33 + 2.0/3.0*ks;

        invariants = calcInvariants(tau_new);

    }

    if( iter == 999 )
    {
        std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizble.";
    }

    return tau_new;

}

matrix9 Plume::invert3(const matrix9& A)
{
    // note that with Bailey's code, the input A.e21, A.e31, and A.e32 are zeros even though they are used here
    // at least when using this on tau to calculate the inverse stress tensor. This is not true when calculating the inverse A matrix
    // for the Ax=b calculation

    // set the output value storage
    matrix9 Ainv;

    // now calculate the determinant
    double det = A.e11*(A.e22*A.e33 - A.e23*A.e32) - A.e12*(A.e21*A.e33 - A.e23*A.e31) + A.e13*(A.e21*A.e32 - A.e22*A.e31);

    // check for near zero value determinants
    if(std::abs(det) < 1e-10)
    {
        det = 10e10;
        std::cout << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
        std::cout << "abs(det) = \"" << std::abs(det) << "\",  A.e11 = \"" << A.e11 << "\", A.e12 = \"" << A.e12 << "\", A.e13 = \"" 
            << A.e13 << "\", A.e21 = \"" << A.e21 << "\", A.e22 = \"" << A.e22 << "\", A.e23 = \"" << A.e23 << "\", A.31 = \"" 
            << A.e31 << "\" A.e32 = \"" << A.e32 << "\", A.e33 = \"" << A.e33 << "\"" << std::endl;
    }

    // calculate the inverse
    Ainv.e11 =  (A.e22*A.e33 - A.e23*A.e32)/det;
    Ainv.e12 = -(A.e12*A.e33 - A.e13*A.e32)/det;
    Ainv.e13 =  (A.e12*A.e23 - A.e22*A.e13)/det;
    Ainv.e21 = -(A.e21*A.e33 - A.e23*A.e31)/det;
    Ainv.e22 =  (A.e11*A.e33 - A.e13*A.e31)/det;
    Ainv.e23 = -(A.e11*A.e23 - A.e13*A.e21)/det;
    Ainv.e31 =  (A.e21*A.e32 - A.e31*A.e22)/det;
    Ainv.e32 = -(A.e11*A.e32 - A.e12*A.e31)/det;
    Ainv.e33 =  (A.e11*A.e22 - A.e12*A.e21)/det;


    return Ainv;
}

vec3 Plume::matmult(const matrix9& Ainv,const vec3& b)
{
    // initialize output
    vec3 x;

    // now calculate the Ax=b x value from the input inverse A matrix and b matrix
    x.e11 = b.e11*Ainv.e11 + b.e21*Ainv.e12 + b.e31*Ainv.e13;
    x.e21 = b.e11*Ainv.e21 + b.e21*Ainv.e22 + b.e31*Ainv.e23;
    x.e31 = b.e11*Ainv.e31 + b.e21*Ainv.e32 + b.e31*Ainv.e33;

    return x;
}


void Plume::setBCfunctions(std::string xBCtype,std::string yBCtype,std::string zBCtype)
{
    // the idea is to use the string input BCtype to determine which boundary condition function to use later in the program, and to have a function pointer
    // point to the required function. I learned about pointer functions from this website: https://www.learncpp.com/cpp-tutorial/78-function-pointers/
    
    std::cout << "xBCtype = \"" << xBCtype << "\"" << std::endl;
    std::cout << "yBCtype = \"" << yBCtype << "\"" << std::endl;
    std::cout << "zBCtype = \"" << zBCtype << "\"" << std::endl;
    if(xBCtype == "exiting")
    {
        enforceWallBCs_x = &Plume::enforceWallBCs_exiting;  // the enforceWallBCs_x pointer function now points to the enforceWallBCs_exiting function
    }else if(xBCtype == "periodic")
    {
        enforceWallBCs_x = &Plume::enforceWallBCs_periodic;  // the enforceWallBCs_x pointer function now points to the enforceWallBCs_periodic function
    }else if(xBCtype == "reflection")
    {
        enforceWallBCs_x = &Plume::enforceWallBCs_reflection;  // the enforceWallBCs_x pointer function now points to the enforceWallBCs_reflection function
    }else
    {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input xBCtype \"" << xBCtype 
            << "\" has not been implemented in code! Available xBCtypes are \"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if(yBCtype == "exiting")
    {
        enforceWallBCs_y = &Plume::enforceWallBCs_exiting;  // the enforceWallBCs_y pointer function now points to the enforceWallBCs_exiting function
    }else if(yBCtype == "periodic")
    {
        enforceWallBCs_y = &Plume::enforceWallBCs_periodic;  // the enforceWallBCs_y pointer function now points to the enforceWallBCs_periodic function
    }else if(yBCtype == "reflection")
    {
        enforceWallBCs_y = &Plume::enforceWallBCs_reflection;  // the enforceWallBCs_y pointer function now points to the enforceWallBCs_reflection function
    }else
    {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input yBCtype \"" << yBCtype 
            << "\" has not been implemented in code! Available yBCtypes are \"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(zBCtype == "exiting")
    {
        enforceWallBCs_z = &Plume::enforceWallBCs_exiting;  // the enforceWallBCs_z pointer function now points to the enforceWallBCs_exiting function
    }else if(zBCtype == "periodic")
    {
        enforceWallBCs_z = &Plume::enforceWallBCs_periodic;  // the enforceWallBCs_z pointer function now points to the enforceWallBCs_periodic function
    }else if(zBCtype == "reflection")
    {
        enforceWallBCs_z = &Plume::enforceWallBCs_reflection;  // the enforceWallBCs_z pointer function now points to the enforceWallBCs_reflection function
    }else
    {
        std::cerr << "!!! Plume::setBCfunctions() error !!! input zBCtype \"" << zBCtype 
            << "\" has not been implemented in code! Available zBCtypes are \"exiting\", \"periodic\", \"reflection\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}



void Plume::enforceWallBCs_exiting(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    // this may change as we figure out the reflections vs depositions on buildings and terrain as well as the outer domain
    // probably will become some kind of inherited function or a pointer function that can be chosen at initialization time
    // for now, if it goes out of the domain, set isActive to false

    if( pos < domainStart || pos > domainEnd )
    {
        isActive = false;
    }
}

void Plume::enforceWallBCs_periodic(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    
    double domainSize = domainEnd - domainStart;
    int loopCountLeft = 0;
    int loopCountRight = 0;
        
    std::cout << "enforceWallBCs_periodic starting pos = \"" << pos << "\", domainStart = \"" << domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;

    if(domainSize != 0)
    {
        while( pos < domainStart )
        {
            pos = pos + domainSize;
            loopCountLeft = loopCountLeft + 1;
        }
        while( pos > domainEnd )
        {
            pos = pos - domainSize;
            loopCountRight = loopCountRight + 1;
        }
    }
    
    std::cout << "enforceWallBCs_periodic ending pos = \"" << pos << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << std::endl;

}

void Plume::enforceWallBCs_reflection(double& pos,double& velPrime,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    if( isActive == true )
    {

        std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velPrime = \"" << velPrime << "\", velFluct_old = \"" <<
                velFluct_old << "\", domainStart = \"" << domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;

        int reflectCount = 0;
        int loopCountLeft = 0;
        int loopCountRight = 0;
        while( pos < domainStart || pos > domainEnd )
        {
            if( pos > domainEnd )
            {
                pos = domainEnd - (pos - domainEnd);
                velPrime = -velPrime;
                velFluct_old = -velFluct_old;
                loopCountLeft = loopCountLeft + 1;
            } else if( pos < domainStart )
            {
                pos = domainStart - (pos - domainStart);
                velPrime = -velPrime;
                velFluct_old = -velFluct_old;
                loopCountRight = loopCountRight + 1;
            }
            reflectCount = reflectCount + 1;

            // if the velocity is so large that the particle would reflect more than 100 times, 
            // the boundary condition could fail.
            if( reflectCount == 10 )    // use 10 since 100 is really expensive right now!
            {
                if( pos > domainEnd )
                {
                    std::cout << "warning (Plume::enforceWallBCs_reflection): upper boundary condition failed! Setting isActive to false. pos = \"" << pos << "\"" << std::endl;
                    isActive = false;
                }
                if( pos < domainStart )
                {
                    std::cout << "warning (Plume::enforceWallBCs_reflection): lower boundary condition failed! Setting isActive to false. xPos = \"" << pos << "\"" << std::endl;
                    isActive = false;
                }
            }
        }   // while outside of domain

        std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velPrime = \"" << velPrime << "\", velFluct_old = \"" <<
                velFluct_old << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << loopCountRight << "\", reflectCount = \"" <<
                reflectCount << "\"" << std::endl;

    }   // if isActive == true
}


void Plume::setFinishedParticleVals(double& xPos,double& yPos,double& zPos, const bool& isActive, const bool& isRogue)
{
    if(isActive == false || isRogue == true)
    {
        xPos = -999.0;
        yPos = -999.0;
        zPos = -999.0;
    }
}


void Plume::save(Output* output)
{
    
    std::cout << "[Plume] \t Saving particle concentrations" << std::endl;
    
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
    for (int i = 0; i < output_scalar_dbl.size(); i++)
    {
        output->saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
    }
    
    // loop through 2D double fields to save
    for (int i = 0; i < output_vector_dbl.size(); i++)
    {

        // x,y,z, terrain saved once with no time component
        if( i < 3 && output_counter == 0 )
        {
            output->saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
        } else
        {
            output->saveField2D(output_vector_dbl[i].name, vector_index,
                                vector_size, *output_vector_dbl[i].data);
        }
    }

    // remove x, y, z, terrain from output array after first save
    if( output_counter == 0 )
    {
        output_vector_dbl.erase(output_vector_dbl.begin(),output_vector_dbl.begin()+3);
    }

    // increment for next time insertion
    output_counter +=1;

}



void Plume::average(const int tStep,const Dispersion* dis, const Urb* urb)
{
    // for all particles see where they are relative to the
    // concentration collection boxes
    for(int i = 0; i < dis->pointList.size(); i++)
    {
        // ????
//        if( tStrt.at(i) > timeStepStamp.at(tStep) )
//        {
//            continue;
//        }
        
        double xPos = dis->pointList.at(i).pos.e11;
        double yPos = dis->pointList.at(i).pos.e21;
        double zPos = dis->pointList.at(i).pos.e31;
        
        if ( (xPos > 0.0 && yPos > 0.0 && zPos > 0.0) &&
             (xPos < (urb->grid.nx*urb->grid.dx)) &&
             (yPos < (urb->grid.ny*urb->grid.dy)) &&
             (zPos < (urb->grid.nz*urb->grid.dz)) ) {
            
            // ????
            if( zPos == -1 )
            {
                continue;
            }

            // Calculate which collection box this particle is currently
            // in

            int iV = int(xPos/urb->grid.dx);
            int jV = int(yPos/urb->grid.dy);
            int kV = int(zPos/urb->grid.dz) + 1;    // why is this + 1 here?
            int idx = (int)((xPos-lBndx)/boxSizeX);
            int idy = (int)((yPos-lBndy)/boxSizeY);
            int idz = (int)((zPos-lBndz)/boxSizeZ);

            if( xPos < lBndx )
            {
                idx = -1;
            }
            if( yPos < lBndy )
            {
                idy = -1;
            }
            if( zPos < lBndz )
            {
                idz = -1;
            }

            int id = 0;
            // if( idx >= 0 && idx < nBoxesX && idy >= 0 && idy < nBoxesY && idz >= 0 && idz < nBoxesZ && tStrt.at(i) <= timeStepStamp.at(tStep) )
            if( idx >= 0 && idx < nBoxesX && idy >= 0 && idy < nBoxesY && idz >= 0 && idz < nBoxesZ )
            {
                id = idz*nBoxesY*nBoxesX + idy*nBoxesX + idx;
                cBox.at(id) = cBox.at(id) + 1.0;
            }
        }
        
    }   // particle loop

}

void Plume::writeSimInfoFile(Dispersion* dis, const double& current_time)
{
    std::string outputFolder = dis->debugOutputFolder;

    // if the debug output folder is an empty string "", the debug output variables won't be written
    if( outputFolder == "" )
    {
        return;
    }

    std::cout << "writing simInfoFile" << std::endl;


    // set some variables for use in the function
    FILE *fzout;    // changing file to which information will be written
    

    // now write out the simulation information to the debug folder
    

    // comment out to choose which saveBasename to use
    std::string saveBasename = "sinewave_HeteroAnisoImplicitTurb";
    //std::string saveBasename = "channel_HeteroAnisoImplicitTurb";
    //std::string saveBasename = "LES_HeteroAnisoImplicitTurb";
    double C_0 = 4.0;

    // add timestep to saveBasename variable
    std::cout << "running to_string on dt to add to saveBasename" << std::endl;
    saveBasename = saveBasename + "_" + std::to_string(dt);


    std::string outputFile = outputFolder + "/sim_info.txt";
    std::cout << "opening simInfoFile for write" << std::endl;
    fzout = fopen(outputFile.c_str(), "w");
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"saveBasename     = %s\n",saveBasename.c_str());
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"C_0              = %lf\n",C_0);
    fprintf(fzout,"timestep         = %lf\n",dt);
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"current_time     = %lf\n",current_time);
    fprintf(fzout,"rogueCount       = %0.0lf\n",dis->isRogueCount);
    fprintf(fzout,"isActiveCount    = %0.0lf\n",dis->isActiveCount);
    fprintf(fzout,"\n");    // a purposeful blank line
    fprintf(fzout,"x_nCells         = %d\n",nx);
    fprintf(fzout,"y_nCells         = %d\n",ny);
    fprintf(fzout,"z_nCells         = %d\n",nz);
    fprintf(fzout,"nParticles       = %d\n",dis->pointList.size());
    fprintf(fzout,"\n");    // a purposeful blank line
    fclose(fzout);


    // now that all is finished, clean up the file pointer
    fzout = NULL;

}