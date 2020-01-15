//
//  Plume.cpp
//  
//  This class handles plume model
//

#include "Plume.hpp"

Plume::Plume(Urb* urb,Dispersion* dis, PlumeInputData* PID, Output* output) {
    
    std::cout<<"[Plume] \t Setting up particles "<<std::endl;
    

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
    
    /* setup the sampling box concentration information */

    sCBoxTime = PID->colParams->timeStart;
    avgTime  = PID->colParams->timeAvg;

    nBoxesX = PID->colParams->nBoxesX;
    nBoxesY = PID->colParams->nBoxesY;
    nBoxesZ = PID->colParams->nBoxesZ;

    lBndx = PID->colParams->boxBoundsX1;
    uBndx = PID->colParams->boxBoundsX2;
    lBndy = PID->colParams->boxBoundsY1;
    uBndy = PID->colParams->boxBoundsY2;
    lBndz = PID->colParams->boxBoundsZ1;
    uBndz = PID->colParams->boxBoundsZ2;
    
    boxSizeX = (uBndx-lBndx)/(nBoxesX);
    boxSizeY = (uBndy-lBndy)/(nBoxesY);
    boxSizeZ = (uBndz-lBndz)/(nBoxesZ);
    
    volume = boxSizeX*boxSizeY*boxSizeZ;
    
    
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);
    zBoxCen.resize(nBoxesZ);
    
    
    int zR = 0;
    int yR = 0;
    int xR = 0;
    for(int k = 0; k < nBoxesZ; ++k)
    {
        zBoxCen.at(k) = lBndz + (zR*boxSizeZ) + (boxSizeZ/2.0);
        zR++;
    }
    for(int j = 0; j < nBoxesY; ++j)
    {
        yBoxCen.at(j) = lBndy + (yR*boxSizeY) + (boxSizeY/2.0);
        yR++;
    }
    for(int i = 0; i <nBoxesX; ++i)
    {
        xBoxCen.at(i) = lBndx + (xR*boxSizeX) + (boxSizeX/2.0);
        xR++;
    }

    cBox.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    conc.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    
    
    /* make copies of important input time variables */

    dt = PID->simParams->timeStep;
    simDur = PID->simParams->simDur;


    // set up time details
    numTimeStep = std::ceil(simDur/dt);
    timeStepStamp.resize(numTimeStep);
    for(int i = 0; i < numTimeStep; ++i)
    {
        timeStepStamp.at(i) = i*dt + dt;
    }
    

    // set additional values from the input
    invarianceTol = PID->simParams->invarianceTol;
    C_0 = PID->simParams->C_0;
    updateFrequency_particleLoop = PID->simParams->updateFrequency_particleLoop;
    updateFrequency_timeLoop = PID->simParams->updateFrequency_timeLoop;
    

    /* setup boundary condition functions */

    // now get the input boundary condition types from the inputs
    std::string xBCtype = PID->BCs->xBCtype;
    std::string yBCtype = PID->BCs->yBCtype;
    std::string zBCtype = PID->BCs->zBCtype;

    // now set the boundary condition function for the plume runs, checking to make sure the input BCtypes are legitimate
    setBCfunctions(xBCtype,yBCtype,zBCtype);
    
    
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

        if( nextSetOfParticles.size() != 0 )
        {
            auto timerEnd_particleRelease = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_particleRelease = timerEnd_particleRelease - timerStart_particleRelease;
            std::cout << "finished emitting \"" << nextSetOfParticles.size() << "\" particles from \"" << dis->allSources.size() 
                    << "\" sources. Total numParticles = \"" << dis->pointList.size() << "\"" << std::endl;
            std::cout << "\telapsed time: " << elapsed_particleRelease.count() << " s" << std::endl;   // Print out elapsed execution time
        }

        // get the isRogue and isActive count from the dispersion class
        double isRogueCount = dis->isRogueCount;
        double isActiveCount = dis->isActiveCount;

        // Move each particle for every time step
        // Advection Loop

        //if( tStep % updateFrequency_timeLoop == 0 || tStep == numTimeStep - 1 )
        //{
            // I want to get an idea of the overall time of the advection loop, and different parts of the advection loop
            auto timerStart_advection = std::chrono::high_resolution_clock::now();
        //}

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

                //if( par % updateFrequency_particleLoop == 0 )
                //{
                    // overall particle timer
                    auto timerStart_particle = std::chrono::high_resolution_clock::now();
                //}


                // this is getting the current position for where the particle is at for a given time
                // if it is the first time a particle is ever released, then the value is already set at the initial value
                double xPos = dis->pointList.at(par).xPos;
                double yPos = dis->pointList.at(par).yPos;
                double zPos = dis->pointList.at(par).zPos;

                // this is the old velFluct value, that will be overwritten during the solver
                // hmm, Bailey's code just starts out setting these values to zero,
                // so the velFluct values are actually the old velFluct. velFluct_old and velFluct are probably identical and kind of redundant in this implementation
                // but it shouldn't hurt anything for now, even if it is redundant
                // besides, it will probably change a bit once I figure out what exactly I want outputted on a regular, and on a debug basis
                double uFluct = dis->pointList.at(par).uFluct;
                double vFluct = dis->pointList.at(par).vFluct;
                double wFluct = dis->pointList.at(par).wFluct;

                // should also probably grab and store the old values in this same way
                // these consist of velFluct_old and tao_old
                // also need to keep track of a delta_velFluct, though delta_velFluct doesn't need grabbed as a value till later now that I think on it
                double uFluct_old = dis->pointList.at(par).uFluct_old;
                double vFluct_old = dis->pointList.at(par).vFluct_old;
                double wFluct_old = dis->pointList.at(par).wFluct_old;
                double txx_old = dis->pointList.at(par).txx_old;
                double txy_old = dis->pointList.at(par).txy_old;
                double txz_old = dis->pointList.at(par).txz_old;
                double tyy_old = dis->pointList.at(par).tyy_old;
                double tyz_old = dis->pointList.at(par).tyz_old;
                double tzz_old = dis->pointList.at(par).tzz_old;
                
                
                /*
                    now get the values for the current iteration
                    will need to use the interp3D function
                    Need to get velMean, CoEps, tao, flux_div
                    Then need to call makeRealizable on tao then calculate inverse tao
                */


                // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
                // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
                eul->setInterp3Dindexing(xPos,yPos,zPos);



                // this is the Co times Eps for the particle
                double CoEps = eul->interp3D(turb->CoEps,"CoEps");
                //double CoEps = eul->interp3D(turb->CoEps,"Eps");
                
                
                // this is the current velMean value
                double uMean = eul->interp3D(urb->u,"velMean");
                double vMean = eul->interp3D(urb->v,"velMean");
                double wMean = eul->interp3D(urb->w,"velMean");
                
                // this is the current reynolds stress tensor
                double txx_before = eul->interp3D(turb->txx,"tau");
                double txy_before = eul->interp3D(turb->txy,"tau");
                double txz_before = eul->interp3D(turb->txz,"tau");
                double tyy_before = eul->interp3D(turb->tyy,"tau");
                double tyz_before = eul->interp3D(turb->tyz,"tau");
                double tzz_before = eul->interp3D(turb->tzz,"tau");
                
                // now need flux_div_dir, not the different dtxxdx type components
                double flux_div_x = eul->interp3D(eul->flux_div_x,"flux_div");
                double flux_div_y = eul->interp3D(eul->flux_div_y,"flux_div");
                double flux_div_z = eul->interp3D(eul->flux_div_z,"flux_div");


                // now need to call makeRealizable on tao
                // directly modifies the values of tao
                double txx = txx_before;
                double txy = txy_before;
                double txz = txz_before;
                double tyy = tyy_before;
                double tyz = tyz_before;
                double tzz = tzz_before;
                makeRealizable(txx,txy,txz,tyy,tyz,tzz);
                
                
                // now need to calculate the inverse values for tao
                // directly modifies the values of tao
                // I just noticed that Bailey's code always leaves the last three components alone, never filled with the symmetrical tensor values
                // this seems fine for makeRealizable, but I wonder if it messes with the invert3 stuff since those values are used even though they are empty in his code
                // going to send in 9 terms anyways to try to follow Bailey's method for now
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
                double xRandn = random::norRan();   // should be randn() matlab equivalent, which is a normally distributed random number
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
                double A_11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*dt;
                double A_12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*dt;
                double A_13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*dt;

                double A_21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*dt;
                double A_22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*dt;
                double A_23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*dt;
                
                double A_31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*dt;
                double A_32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*dt;
                double A_33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*dt;


                double b_11 = -uFluct_old - 0.50*flux_div_x*dt - std::sqrt(CoEps*dt)*xRandn;
                double b_21 = -vFluct_old - 0.50*flux_div_y*dt - std::sqrt(CoEps*dt)*yRandn;
                double b_31 = -wFluct_old - 0.50*flux_div_z*dt - std::sqrt(CoEps*dt)*zRandn;


                // now prepare for the Ax=b calculation by calculating the inverted A matrix
                // directly modifies the values of the A matrix
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
                matmult(A_11_inv,A_12_inv,A_13_inv,A_21_inv,A_22_inv,A_23_inv,A_31_inv,A_32_inv,A_33_inv,b_11,b_21,b_31, uFluct,vFluct,wFluct);
                

                // now check to see if the value is rogue or not
                if( ( std::abs(uFluct) >= vel_threshold || isnan(uFluct) ) && nx > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible uFluct was \"" << uFluct << "\"" << std::endl;

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

                    uFluct = 0.0;
                    isRogue = true;
                }
                if( ( std::abs(vFluct) >= vel_threshold || isnan(vFluct) ) && ny > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible vFluct was \"" << vFluct << "\"" << std::endl;

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

                    vFluct = 0.0;
                    isRogue = true;
                }
                if( ( std::abs(wFluct) >= vel_threshold || isnan(wFluct) ) && nz > 1 )
                {
                    std::cout << "Particle # " << par << " is rogue." << std::endl;
                    std::cout << "responsible wFluct was \"" << wFluct << "\"" << std::endl;

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

                    wFluct = 0.0;
                    isRogue = true;
                }

                // Do you need this???
                // ONLY if this should never happen....
                //    assert( isRogue == false );
                // maybe implement this after the thesis work. Currently use it to know if something is going wrong

                
                // now update the particle position for this iteration
                // at some point in time, need to do a CFL condition for only moving one eulerian grid cell at a time
                // as well as a separate CFL condition for the particle timestep
                // this would mean adding some kind of while loop, with an adaptive timestep, controlling the end of the while loop
                //  with the sampling time increment. This means all particles can do multiple time iterations, each with their own timestep
                // but at the sampling timestep, particles are allowed to catch up so they are all calculated by that time
                // currently, we use the sampling timestep so we may violate the eulerian grid CFL condition. But is simpler to work with when getting started
                double disX = (uMean + uFluct)*dt;
                double disY = (vMean + vFluct)*dt;
                double disZ = (wMean + wFluct)*dt;
                
                xPos = xPos + disX;
                yPos = yPos + disY;
                zPos = zPos + disZ;


                //std::cout << "applying wallBC" << std::endl;
                //auto timerStart_wallBC = std::chrono::high_resolution_clock::now();

                // now apply boundary conditions
                // notice that this is the old fashioned style for calling a pointer function
                (this->*enforceWallBCs_x)(xPos,uFluct,uFluct_old,isActive, domainXstart,domainXend);
                (this->*enforceWallBCs_y)(yPos,vFluct,vFluct_old,isActive, domainYstart,domainYend);
                (this->*enforceWallBCs_z)(zPos,wFluct,wFluct_old,isActive, domainZstart,domainZend);
                
                // now set the particle values for if they are rogue or outside the domain
                setFinishedParticleVals(xPos,yPos,zPos, isActive,isRogue);

                
                //auto timerEnd_wallBC = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> elapsed_wallBC = timerEnd_wallBC - timerStart_wallBC;
                //std::cout << "wallBC for particle par[" << par << "] and timeStepStamp[" << timeStepStamp.at(tStep) << "] finished" << std::endl;
                //std::cout << "\telapsed time: " << elapsed_wallBC.count() << " s" << std::endl;   // Print out elapsed execution time



                // now update the old values and current values in the dispersion storage to be ready for the next iteration
                // also calculate the velFluct increment
                // this is extremely important for output and the next iteration to work correctly
                dis->pointList.at(par).uFluct = uFluct;
                dis->pointList.at(par).vFluct = vFluct;
                dis->pointList.at(par).wFluct = wFluct;
                dis->pointList.at(par).xPos = xPos;
                dis->pointList.at(par).yPos = yPos;
                dis->pointList.at(par).zPos = zPos;

                dis->pointList.at(par).delta_uFluct = uFluct - uFluct_old;
                dis->pointList.at(par).delta_vFluct = vFluct - vFluct_old;
                dis->pointList.at(par).delta_wFluct = wFluct - wFluct_old;
                dis->pointList.at(par).uFluct_old = uFluct;
                dis->pointList.at(par).vFluct_old = vFluct;
                dis->pointList.at(par).wFluct_old = wFluct;

                dis->pointList.at(par).txx_old = txx;
                dis->pointList.at(par).txy_old = txy;
                dis->pointList.at(par).txz_old = txz;
                dis->pointList.at(par).tyy_old = tyy;
                dis->pointList.at(par).tyz_old = tyz;
                dis->pointList.at(par).tzz_old = tzz;

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


                if(  ( (tStep+1) % updateFrequency_timeLoop == 0 || tStep == 0 || tStep == numTimeStep-1 ) && ( par % updateFrequency_particleLoop == 0 || par == 0 || par == dis->pointList.size()-1 )  )
                {
                    auto timerEnd_particle = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_particle = timerEnd_particle - timerStart_particle;
                    std::cout << "particle iteration par[" << par << "] finished. timestep = \"" << timeStepStamp.at(tStep) << "\"" << std::endl;
                    std::cout << "\telapsed time: " << elapsed_particle.count() << " s" << std::endl;   // Print out elapsed execution time
                }
            
            }   // if isActive == true and isRogue == false


        } // for (int par=0; par<parToMove;par++)

        // set the isRogueCount for the time iteration in the disperion data
        // hm, I'm almost wondering if this needs to go into dispersion, could just be kept locally,
        // but declared outside the loop to preserve the value. Depends on the requirements for output and debugging
        dis->isRogueCount = isRogueCount;
        dis->isActiveCount = isActiveCount;


        if( (tStep+1) % updateFrequency_timeLoop == 0 || tStep == 0 || tStep == numTimeStep-1 )
        {
            auto timerEnd_advection = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_advection = timerEnd_advection - timerStart_advection;
            std::cout << "advection loop for time = \"" << timeStepStamp.at(tStep) << "\" (timeStepStamp[" << timeStepStamp.at(tStep) << "]) finished" << std::endl;
            std::cout << "\telapsed time: " << elapsed_advection.count() << " s" << std::endl;   // Print out elapsed execution time
        }


        
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

        if( (tStep+1) % updateFrequency_timeLoop == 0 || tStep == 0 || tStep == numTimeStep-1 )
        {
            std::cout << "time = \"" << timeStepStamp.at(tStep) << "\", isRogueCount = \"" << dis->isRogueCount << "\", isActiveCount = \"" << dis->isActiveCount << "\"" << std::endl;
        }


        // 
        // For all particles that need to be removed from the particle
        // advection, remove them now
        //
        // Purge the advection list of all the unneccessary particles....
        // 
        // for now I want to keep them for the thesis work information and debugging

    } // for(tStep=0; tStep<numTimeStep; tStep++)

    auto timerEnd_timeIntegration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_timeIntegration = timerEnd_timeIntegration - timerStart_timeIntegration;
    std::cout << "time integration loop finished" << std::endl;
    std::cout << "\telapsed time: " << elapsed_timeIntegration.count() << " s" << std::endl;   // Print out elapsed execution time

    // if the debug output folder is an empty string "", the debug output variables won't be written
    writeSimInfoFile(dis,timeStepStamp.at(numTimeStep-1));
    dis->outputVarInfo_text();

}


void Plume::calcInvariants(const double& txx,const double& txy,const double& txz,const double& tyy,const double& tyz,const double& tzz,
                            double& invar_xx,double& invar_yy,double& invar_zz)
{
    // since the x doesn't depend on itself, can just set the output without doing any temporary variables

    // this is just a copy of what is done in Bailey's code
    invar_xx = txx + tyy + tzz;

    invar_yy = txx*tyy + txx*tzz + tyy*tzz - txy*txy - txz*txz - tyz*tyz;

    invar_zz = txx*(tyy*tzz - tyz*tyz) - txy*(txy*tzz - tyz*txz) + txz*(txy*tyz - tyy*txz);
}

void Plume::makeRealizable(double& txx,double& txy,double& txz,double& tyy,double& tyz,double& tzz)
{
    // first calculate the invariants and see if they are already realizable
    // the calcInvariants function modifies the values directly, so they always need initialized to something before being sent into said function to be calculated

    double invar_xx = 0.0;
    double invar_yy = 0.0;
    double invar_zz = 0.0;
    calcInvariants(txx,txy,txz,tyy,tyz,tzz,  invar_xx,invar_yy,invar_zz);
    if( invar_xx > invarianceTol && invar_yy > invarianceTol && invar_zz > invarianceTol )
    {
        //std::cout << "tau already realizable" << std::endl;
        return;     // tau is already realizable
    }

    // since tau is not already realizable, need to make it realizeable
    // start by making a guess of ks, the subfilter scale tke
    // I keep wondering if we can use the input Turb->tke for this or if we should leave it as is
    double b = 4.0/3.0*(txx + tyy + tzz);   // I think this is 4.0/3.0*invar_xx as well
    double c = txx*tyy + txx*tzz + tyy*tzz - txy*txy - txz*txz - tyz*tyz;   // this is probably invar_yy
    double ks = 1.01*(-b + std::sqrt(b*b - 16.0/3.0*c)) / (8.0/3.0);

    // if the initial guess is bad, use the straight up invariance.e11 value
    if( ks < invarianceTol || isnan(ks) )
    {
        ks = 0.5*std::abs(txx + tyy + tzz);  // looks like 0.5*abs(invar_xx)
    }

    // now set the initial values of the new make realizable tau, 
    // since the function modifies tau directly, so tau already exists, can skip this step and just do what is already done inside the loop
    // notice that through all this process, only the diagonals are really increased by a value of 0.05% of the subfilter tke ks
    // had to watch carefully, but for once don't need temporary values as there are no multi line dependencies 
    //  from one component of tau to another the way this is written
    
    // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till tau is realizable
    // or if too many iterations go on, give a warning.
    // I've had trouble with this taking too long
    //  if it isn't realizable, so maybe another approach for when the iterations are reached might be smart
    int iter = 0;
    while( (invar_xx < invarianceTol || invar_yy < invarianceTol || invar_zz < invarianceTol) && iter < 1000 )
    {
        iter = iter + 1;

        ks = ks*1.050;      // increase subfilter tke by 5%

        txx = txx + 2.0/3.0*ks;
        tyy = tyy + 2.0/3.0*ks;
        tzz = tzz + 2.0/3.0*ks;

        calcInvariants(txx,txy,txz,tyy,tyz,tzz,  invar_xx,invar_yy,invar_zz);

    }

    if( iter == 999 )
    {
        std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizble.";
    }

}

void Plume::invert3(double& A_11,double& A_12,double& A_13,double& A_21,double& A_22,double& A_23,double& A_31,double& A_32,double& A_33)
{
    // note that with Bailey's code, the input A_21, A_31, and A_32 are zeros even though they are used here
    // at least when using this on tau to calculate the inverse stress tensor. This is not true when calculating the inverse A matrix
    // for the Ax=b calculation


    // now calculate the determinant
    double det = A_11*(A_22*A_33 - A_23*A_32) - A_12*(A_21*A_33 - A_23*A_31) + A_13*(A_21*A_32 - A_22*A_31);

    // check for near zero value determinants
    if(std::abs(det) < 1e-10)
    {
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

void Plume::matmult(const double& A_11,const double& A_12,const double& A_13,const double& A_21,const double& A_22,const double& A_23,
                    const double& A_31,const double& A_32,const double& A_33,const double& b_11,const double& b_21,const double& b_31,
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



void Plume::enforceWallBCs_exiting(double& pos,double& velFluct,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    // this may change as we figure out the reflections vs depositions on buildings and terrain as well as the outer domain
    // probably will become some kind of inherited function or a pointer function that can be chosen at initialization time
    // for now, if it goes out of the domain, set isActive to false

    if( pos < domainStart || pos > domainEnd )
    {
        isActive = false;
    }
}

void Plume::enforceWallBCs_periodic(double& pos,double& velFluct,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    
    double domainSize = domainEnd - domainStart;
    int loopCountLeft = 0;
    int loopCountRight = 0;

    /*    
    std::cout << "enforceWallBCs_periodic starting pos = \"" << pos << "\", domainStart = \"" << domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;
    */

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
    
    /*
    std::cout << "enforceWallBCs_periodic ending pos = \"" << pos << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << std::endl;
    */

}

void Plume::enforceWallBCs_reflection(double& pos,double& velFluct,double& velFluct_old,bool& isActive, const double& domainStart,const double& domainEnd)
{
    if( isActive == true )
    {

        /*
        std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velFluct = \"" << velFluct << "\", velFluct_old = \"" <<
                velFluct_old << "\", domainStart = \"" << domainStart << "\", domainEnd = \"" << domainEnd << "\"" << std::endl;
        */

        int reflectCount = 0;
        int loopCountLeft = 0;
        int loopCountRight = 0;
        while( pos < domainStart || pos > domainEnd )
        {
            if( pos > domainEnd )
            {
                pos = domainEnd - (pos - domainEnd);
                velFluct = -velFluct;
                velFluct_old = -velFluct_old;
                loopCountLeft = loopCountLeft + 1;
            } else if( pos < domainStart )
            {
                pos = domainStart - (pos - domainStart);
                velFluct = -velFluct;
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
                break;
            }
        }   // while outside of domain

        /*
        std::cout << "enforceWallBCs_reflection starting pos = \"" << pos << "\", velFluct = \"" << velFluct << "\", velFluct_old = \"" <<
                velFluct_old << "\", loopCountLeft = \"" << loopCountLeft << "\", loopCountRight = \"" << loopCountRight << "\", reflectCount = \"" <<
                reflectCount << "\"" << std::endl;
        */

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
        
        double xPos = dis->pointList.at(i).xPos;
        double yPos = dis->pointList.at(i).yPos;
        double zPos = dis->pointList.at(i).zPos;
        
        if ( (xPos > 0.0 && yPos > 0.0 && zPos > 0.0) &&
             (xPos < (nx*dx)) && (yPos < (ny*dy)) && (zPos < (nz*dz)) ) {
            
            // ????
            if( zPos == -1 )
            {
                continue;
            }

            // Calculate which collection box this particle is currently
            // in

            int iV = int(xPos/dx);
            int jV = int(yPos/dy);
            int kV = int(zPos/dz) + 1;    // why is this + 1 here? Probably is assuming a periodic grid for some dimensions, or to capture a right hand side wall? no idea
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
            // if( idx >= 0 && idx < nBoxesX && idy >= 0 && idy < nBoxesY && idz >= 0 && idz < nBoxesZ && dis->pointList.at(i).tStrt <= timeStepStamp.at(tStep) )
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