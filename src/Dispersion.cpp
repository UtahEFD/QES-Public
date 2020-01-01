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
#include "SourceUniformDomain.hpp"

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
    vec3 ptSourcePos;
    ptSourcePos.e11 = 40.0;
    ptSourcePos.e21 = 80.0;
    ptSourcePos.e31 = 30.0;
    SourceKind *sPtr0 = new SourcePoint( ptSourcePos, 100000, ParticleReleaseType::instantaneous, 
                                         domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr0 );
#endif


#if 0
    vec3 pt0, pt1;
    pt0.e11 = 25.0;
    pt0.e21 = 175.0;
    pt0.e31 = 40.0;

    pt1.e11 = 50.0;
    pt1.e21 = 25.0;
    pt1.e31 = 40.0;
    SourceKind *sPtr = new SourceLine( pt0, pt1, 100000, ParticleReleaseType::instantaneous, 
                                       domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );
    allSources.push_back( sPtr );
#endif

#if 1
    // Uniform test case Source
    // This will causes a segfault in Eulerian::interp3D due to kk +
    // kkk being too large
    // 
    // SourceKind *sPtr = new SourceUniformDomain( urb->grid.nx, urb->grid.ny, urb->grid.nz, urb->grid.dx, urb->grid.dy, urb->grid.dz, 100000, 
    //                                             domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend
    //                                             0 ); // last 0 is the fudge factor to force overloading the right constructor

    // Otherwise, this will work
    //SourceKind *sPtr = new SourceUniformDomain( 0.5, 0.5, 0.5, 199.5, 199.5, 199.5, 100000, 
    //                                            domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );

    // alternative and more appropriate form
    SourceKind *sPtr = new SourceUniformDomain( domainXstart, domainYstart, domainZstart, domainXend, domainYend, domainZend, 100000, 
                                                domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend );

    allSources.push_back( sPtr );
#endif

    // ////////////////////
    
    
#if 0
    // this function goes through each source and adds to the particle list the initial positions and release times
    // this function is an initialization factory calling an add source for each type of source
    // eventually this all needs moved to be a single virtual function inside of sourceKind that is overloaded by each source
    // then the if statements in the initialization factory function will go away
    // I think the new virtual function would need to be called something else though, like generatePointInfo()
    addSources(PID->sources);

    // now that all the sources have added to the particle list, it's time to setup the initial values for each particle using their initial positions
    setParticleVals(turb, eul);

    // now sort the big fat list of particles by time, so we can disperse them easily in the same order
    // also so that we can calculate the parPerTimestep values
    sortParticleValsByTime();

    // this function requires the values to be sorted to work correctly, but it goes through finding when the times change,
    // using the changes to store a number of particles to release for each timestep
    calc_parPerTimestep();
#endif
    

    // set the isRogueCount to zero
    isRogueCount = 0.0;
    isActiveCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*sqrt(maxval(turb->sig));  // might need to write a maxval function, since it has to get the largest value from the entire sig array

    
    /*
      Checking if the starting time for the last particle is equal to the duration of
      the simulation (for continous release ONLY)
      would need to modify this to check all particles, probably better to just do the check at the end of each source, 
      so it is known which source causes the problem
    */
#if 0
    if ( fabs(pointList.back().tStrt - simDur) > EPSILON )
    {
        std::cerr << " Error, in start time of the particles" << std::endl;
        exit(1);
    }
#endif

    // set the debug variable output folder
    debugOutputFolder = debugOutputFolder_val;
    
}


void Dispersion::addSources(Sources* sources)
{
    std::cerr << "Dispersion::addSources should NOT BE CALLED!" << std::endl;
    exit(EXIT_FAILURE);

    // first get the number of sources from the sources variable
    int numSources = sources->numSources;

    
    for(int i = 0; i < numSources; i++)
    {
        // all this should work even if there is no points list output, from what I can tell
        
        // get the position and time info for the current source
        // because of polymorphic inheritance, this virtual function should work regardless of the sourceKind
        // the resulting list of points will vary for each source kind
        std::vector<particle> currentPointsList = sources->sources.at(i)->outputPointInfo(dt,simDur);

        // set the number of particles for the current source
        // keeping all these numParticle stuff seperate makes it easier to update the overall point information
        // with the current point information from the current source
        int currentNumPar = currentPointsList.size();

        // update the overall list number of particles
        // keep track of the old number of values so it is easy to update the list in the right way
        // for the first source, oldNumPar will come out as 0.
        int oldNumPar = numPar;
        numPar = numPar + currentNumPar;

        // now resize the overall list number of particles
        // hopefully this works to not get rid of the old values set in the old size locations!
        pointList.resize(numPar);

        // now take the point list information output from the source, and pack it into the overall point list
        for(int i = 0; i < currentNumPar; i++)
        {

            // update the position info
            pointList.at(oldNumPar+i).pos.e11 = currentPointsList.at(i).pos.e11;
            pointList.at(oldNumPar+i).pos.e21 = currentPointsList.at(i).pos.e21;
            pointList.at(oldNumPar+i).pos.e31 = currentPointsList.at(i).pos.e31;

            // now update the time info
            pointList.at(oldNumPar+i).tStrt = currentPointsList.at(i).tStrt;

        }

    }

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

void Dispersion::sortParticleValsByTime()
{
    // this is NOT going to be the most efficient sorting algorythm
    // my idea is to make a copy of the original values, and make a list of the original times
    // also a list of the new indices for the new particles for moving from the original container to the new container
    // since the end goal is to fill the actual container, going to do all the sorting on the copy of said container

    // create the required containers of values
    std::vector<particle> pointListCopy;
    std::vector<double> sortedTimes;
    std::vector<int> sortedIndices;

    // now resize the temporary containers
    pointListCopy.resize(numPar);
    sortedTimes.resize(numPar);
    sortedIndices.resize(numPar);


    // now fill the temporary containers with copies of the original values
    for(int i = 0; i < numPar; i++)
    {
        pointListCopy.at(i) = pointList.at(i);  // without a proper copy operator, will this work correctly? If not, just break it down into a single copy for each freaking part
        sortedTimes.at(i) = pointList.at(i).tStrt;
        sortedIndices.at(i) = i;
    }


    // now go through each copy of the times and the indices, and sort
    // seems like I just need one loop over the second to the last value, but then an inner while loop
    // that goes while the current value is smaller than the value before it or till the value is the first in the list
    // hm, the tricky part is then how to move the values in the containers. I guess technically you don't need a copy 
    // of the whole container to be sorted this way. Instead, when a current value is found to be smaller than the last value
    // a temporary storage of the current and last value can be made, then the values can be swapped in the overall container
    // nice this gets rid of extra storage problems! Except that I have a LOT of values to sort than just the ones that are being
    // sorted. I want to sort all the values by time, it might be smarter to sort a copy of the times and a set of indices for each time
    // then a copy of the whole container can fill the original container using the list of sorted indices
    for(int i = 1; i < numPar; i++)
    {
        // set the values needed for the while loop check
        // these also act as the temporary time storage so the swap is quick and easy in the sorting container
        int backCount = i;
        double current_time = sortedTimes.at(backCount);
        double previous_time = sortedTimes.at(backCount-1);
        while( backCount > 0 && current_time < previous_time )
        {
            // current_time and previous_time need to swap places in the sorting container
            sortedTimes.at(backCount) = previous_time;
            sortedTimes.at(backCount-1) = current_time;

            // set temporary storage of the indices for that swap
            int current_index = sortedIndices.at(backCount);
            int previous_index = sortedIndices.at(backCount-1);

            // now the current and previous indices need to swap places in the index sorting container
            sortedIndices.at(backCount) = previous_index;
            sortedIndices.at(backCount-1) = current_index;

            // update the values needed for the while loop check
            backCount = backCount - 1;
            current_time = sortedTimes.at(backCount);
            // have to watch out for referencing outside the array, unfortunately
            if(backCount != 0)
            {
                previous_time = sortedTimes.at(backCount-1);
            }

        }
    }

    // now I have a list of sorted indices and times and the original and a copy of the original values
    // technically I don't need the sorted times anymore, just the indices. I can use the indices and the copy of the original values
    // to place values from the copy into the original values using the indices to know where everything goes
    for(int i = 0; i < numPar; i++)
    {
        pointList.at(i) = pointListCopy.at(sortedIndices.at(i));    // will this work without a copy constructor in pointList? if not, can expand for each value type
    }

}


void Dispersion::calc_parPerTimestep()
{
    // with the particles sorted by time, go through the list of particle times, and at each particle where the release time
    // exceeds the current timestep, pushback that number of particles to the particles to release per timestep
    
    // first resize the parPerTimestep to be the number of simulation timesteps
    parPerTimestep.resize(numTimeStep);

    int timeStepCount = 0;  // this will keep track of when the timestep has changed
    int cumulatedParticles = 0;     // this is also the already counted particles. Also useful for checking to make sure there isn't a problem with the algorythm
    for(int i = 0; i < numPar; i++)
    {
        if( i == numPar-1 )
        {
            // use the current number of particles and the number of particles already counted to get the particles per this timestep
            parPerTimestep.at(timeStepCount) = i - cumulatedParticles + 1;

            // set the cumulated particles for checking at the end if there was a problem with the algorythm
            cumulatedParticles = cumulatedParticles + parPerTimestep.at(timeStepCount);

            // now update the timeStepCount
            timeStepCount = timeStepCount + 1;
            
        } else if( pointList.at(i).tStrt > timeStepStamp.at(timeStepCount) )
        {
            // use the current number of particles and the number of particles already counted to get the particles per this timestep
            parPerTimestep.at(timeStepCount) = i - cumulatedParticles;

            // set the cumulated particles for checking at the end if there was a problem with the algorythm
            cumulatedParticles = cumulatedParticles + parPerTimestep.at(timeStepCount);

            // now update the timeStepCount
            timeStepCount = timeStepCount + 1;
        }
    }

    if( cumulatedParticles != numPar)
    {
        std::cerr << "Disperion::calc_parPerTimestep Error!" << std::endl;
        std::cerr << "cumulatedParticles not equal to numPar!" << std::endl;
        std::cerr << "cumulatedParticles = \"" << cumulatedParticles << "\", numPar = \"" << numPar << "\"" << std::endl;
        std::cerr << "ENDING PROGRAM!!!" << std::endl;
        exit(1);
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
