//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include <fstream>
#include "Dispersion.h"

#define EPSILON 0.00001

#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

Dispersion::Dispersion(Urb* urb, Turb* turb, PlumeInputData* PID, Eulerian* eul, const std::string& debugOutputFolder_val)
    : pointList(0)
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

    // get the urb domain start and end values, needed for source position range checking
    domainXstart = urb->domainXstart;
    domainXend = urb->domainXend;
    domainYstart = urb->domainYstart;
    domainYend = urb->domainYend;
    domainZstart = urb->domainZstart;
    domainZend = urb->domainZend;


    // have to set the value to something before filling it
    numPar = 0;

    // set up time details
    numTimeStep = std::ceil(simDur/dt);
    timeStepStamp.resize(numTimeStep);
    for(int i = 0; i < numTimeStep; ++i)
    {
        timeStepStamp.at(i) = i*dt + dt;
    }

    // ////////////////////
    // HARD CODE SOME SOURCE TO TEST...
    // test out 1 point source at this location... with these rates
    // and this number of particles... 
#if 0
    double ptSource_xPos = 40.0;
    double ptSource_yPos = 80.0;
    double ptSource_zPos = 30.0;
    SourceKind *sPtr0 = new SourcePoint( ptSource_xPos, ptSource_yPos, ptSource_zPos, 100000, ParticleReleaseType::instantaneous, 
                                         domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr0 );
#endif


#if 0
    double lineSource_xPos0 = 25.0;
    double lineSource_yPos0 = 175.0;
    double lineSource_zPos0 = 40.0;

    double lineSource_xPos1 = 50.0;
    double lineSource_yPos1 = 25.0;
    double lineSource_zPos1 = 40.0;
    SourceKind *sPtr1 = new SourceLine( lineSource_xPos0, lineSource_yPos0, lineSource_zPos0, lineSource_xPos1, lineSource_yPos1, lineSource_zPos1, 
                                       100000, ParticleReleaseType::instantaneous, 
                                       domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr1 );
#endif

#if 0
    double cubeSource_xMin = 0.5;
    double cubeSource_yMin = 0.5;
    double cubeSource_zMin = 0.5;
    double cubeSource_xMax = 199.5;
    double cubeSource_yMax = 199.5;
    double cubeSource_zMax = 199.5;
    SourceKind *sPtr2 = new SourceCube( cubeSource_xMin, cubeSource_yMin, cubeSource_zMin, cubeSource_xMax, cubeSource_yMax, cubeSource_zMax, 
                                        100000, ParticleReleaseType::instantaneous, 
                                        domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr2 );
#endif

#if 0
    SourceKind *sPtr3 = new SourceFullDomain( 100000, ParticleReleaseType::instantaneous, 
                                              domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr3 );
#endif

    // ////////////////////
    

    // get sources from input data and add them to the allSources vector
    // this also calls the check metadata function for the input sources before adding them to the list.
    // the check metadata function should already have been called for all the other sources during the specialized constructor phases used to create them.
    getInputSources(PID);


    // set the isRogueCount to zero
    isRogueCount = 0.0;
    isActiveCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*sqrt(maxval(turb->sig));  // might need to write a maxval function, since it has to get the largest value from the entire sig array


    // set the debug variable output folder
    debugOutputFolder = debugOutputFolder_val;
    
}


void Dispersion::setParticleVals(Turb* turb, Eulerian* eul, std::vector<particle> &newParticles)
{
    // at this time, should be a list of each and every particle that exists
    // might need to vary this to allow for adding sources after the initial constructor
    for(int pIdx=0; pIdx<newParticles.size(); pIdx++)
    {
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(newParticles.at(pIdx).pos);
    
        double rann = random::norRan();   // almost didn't see it, but it does use different random numbers for each direction

        // get the sigma values from the Eulerian grid for the particle value
        diagonal current_sig = eul->interp3D(turb->sig,"sigma2");

        // now set the initial velocity fluctuations for the particle
        newParticles.at(pIdx).prime.e11 = sqrt(current_sig.e11) * rann;  // set the values for the source positions for each particle. Might need to add sqrt of the variance to match Brian's code
        rann=random::norRan();
        newParticles.at(pIdx).prime.e21 = sqrt(current_sig.e22) * rann;
        rann=random::norRan();
        newParticles.at(pIdx).prime.e31 = sqrt(current_sig.e33) * rann;

        // set the initial values for the old stuff
        newParticles.at(pIdx).prime_old.e11 = newParticles.at(pIdx).prime.e11;
        newParticles.at(pIdx).prime_old.e21 = newParticles.at(pIdx).prime.e21;
        newParticles.at(pIdx).prime_old.e31 = newParticles.at(pIdx).prime.e31;

        // get the tau values from the Eulerian grid for the particle value
        matrix6 current_tau = eul->interp3D(turb->tau);

        // set tau_old to the interpolated values for each position
        newParticles.at(pIdx).tau_old.e11 = current_tau.e11;
        newParticles.at(pIdx).tau_old.e12 = current_tau.e12;
        newParticles.at(pIdx).tau_old.e13 = current_tau.e13;
        newParticles.at(pIdx).tau_old.e22 = current_tau.e22;
        newParticles.at(pIdx).tau_old.e23 = current_tau.e23;
        newParticles.at(pIdx).tau_old.e33 = current_tau.e33;

        // set delta_prime to zero for now
        newParticles.at(pIdx).delta_prime.e11 = 0.0;
        newParticles.at(pIdx).delta_prime.e21 = 0.0;
        newParticles.at(pIdx).delta_prime.e31 = 0.0;

        // set isRogue to false and isActive to true for each particle
        newParticles.at(pIdx).isRogue = false;
        newParticles.at(pIdx).isActive = true;
        
    }

}


void Dispersion::getInputSources(PlumeInputData* PID)
{
    int numSources_Input = PID->sources->sources.size();

    if( numSources_Input == 0 )
    {
        std::cerr << "ERROR (Dispersion::getInputSources): there are no sources in the input file!" << std::endl;
        exit(1);
    }

    for(auto sidx=0u; sidx < numSources_Input; sidx++)
    {
        // first create the pointer to the input source
        SourceKind *sPtr;

        // now point the pointer at the source
        sPtr = PID->sources->sources.at(sidx);
        

        // now do anything that is needed to the source via the pointer
        sPtr->checkMetaData(domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend);


        // now add the pointer that points to the source to the list of sources in dispersion
        allSources.push_back( sPtr );
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

void Dispersion::outputVarInfo_text()
{
    // if the debug output folder is an empty string "", the debug output variables won't be written
    if( debugOutputFolder == "" )
    {
        return;
    }

    std::cout << "writing Lagrangian debug variables" << std::endl;

    // set some variables for use in the function
    FILE *fzout;    // changing file to which information will be written
    std::string currentFile = "";
    

    // now write out the Lagrangian grid information to the debug folder
    // at some time this could be wrapped up into a bunch of functions, for now just type it all out without functions


    // make a variable to keep track of the number of particles
    int nPar = pointList.size();
    


    currentFile = debugOutputFolder + "/particle_txx_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e11);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_txy_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e12);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_txz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e13);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tyy_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e22);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tyz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e23);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tzz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tau_old.e33);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_uFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime_old.e11);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_vFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime_old.e21);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_wFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime_old.e31);
    }
    fclose(fzout);



    currentFile = debugOutputFolder + "/particle_uFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime.e11);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_vFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime.e21);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_wFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).prime.e31);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_delta_uFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_prime.e11);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_delta_vFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_prime.e21);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_delta_wFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_prime.e31);
    }
    fclose(fzout);



    // note that isActive is not the same as isRogue, but the expected output is isRogue
    currentFile = debugOutputFolder + "/particle_isActive.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).isRogue);
    }
    fclose(fzout);



    currentFile = debugOutputFolder + "/particle_xPos.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).pos.e11);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_yPos.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).pos.e21);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_zPos.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).pos.e31);
    }
    fclose(fzout);


    // now that all is finished, clean up the file pointer
    fzout = NULL;

}
