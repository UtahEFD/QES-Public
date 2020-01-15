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
    : pointList(0)  // ???
{
    std::cout<<"[Dispersion] \t Setting up sources "<<std::endl;
    

    // get the domain start and end values, needed for source position range checking
    determineDomainSize(urb,turb);
 

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


    // set the isRogueCount and isActiveCount to zero
    isRogueCount = 0.0;
    isActiveCount = 0.0;

    // calculate the threshold velocity
    vel_threshold = 10.0*std::sqrt(getMaxVariance(turb->sig_x,turb->sig_y,turb->sig_z));


    // set the debug variable output folder
    debugOutputFolder = debugOutputFolder_val;
    
}


void Dispersion::determineDomainSize(Urb* urb, Turb* turb)
{

    // multiple ways to do this for now. Could just use the turb grid,
    //  or could determine which grid has the smallest and largest value,
    //  or could use input information to determine if they are cell centered or
    //  face centered and use the dx type values to determine the real domain size.
    // We had a discussion that because there are ghost cells on the grid, probably can
    //  just pretend the ghost cells are a halo region that can still be used during plume solver
    //  but ignored when determining output. This could allow particles to reenter the domain as well.

    // for now, I'm just going to use the urb grid, as having differing grid sizes requires extra info for the interp functions
    domainXstart = urb->urbXstart;
    domainXend = urb->urbXend;
    domainYstart = urb->urbYstart;
    domainYend = urb->urbYend;
    domainZstart = urb->urbZstart;
    domainZend = urb->urbZend;
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


void Dispersion::setParticleVals(Turb* turb, Eulerian* eul, std::vector<particle>& newParticles)
{
    // at this time, should be a list of each and every particle that exists at the given time
    // particles and sources can potentially be added to the list elsewhere
    for(int pIdx=0; pIdx<newParticles.size(); pIdx++)
    {
        // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
        // then get interpolated values from the Eulerian grid to the particle Lagrangian values for multiple datatypes
        eul->setInterp3Dindexing(newParticles.at(pIdx).xPos,newParticles.at(pIdx).yPos,newParticles.at(pIdx).zPos);
    

        // almost didn't see it, but it does use different random numbers for each direction
        double rann = random::norRan();

        // get the sigma values from the Eulerian grid for the particle value
        double current_sig_x = eul->interp3D(turb->sig_x,"sigma2");
        double current_sig_y = eul->interp3D(turb->sig_y,"sigma2");
        double current_sig_z = eul->interp3D(turb->sig_z,"sigma2");

        // now set the initial velocity fluctuations for the particle
        // The  sqrt of the variance is to match Bailey's code
        newParticles.at(pIdx).uFluct = std::sqrt(current_sig_x) * rann;
        rann=random::norRan();      // should be randn() matlab equivalent, which is a normally distributed random number
        newParticles.at(pIdx).vFluct = std::sqrt(current_sig_y) * rann;
        rann=random::norRan();
        newParticles.at(pIdx).wFluct = std::sqrt(current_sig_z) * rann;

        // set the initial values for the old velFluct values
        newParticles.at(pIdx).uFluct_old = newParticles.at(pIdx).uFluct;
        newParticles.at(pIdx).vFluct_old = newParticles.at(pIdx).vFluct;
        newParticles.at(pIdx).wFluct_old = newParticles.at(pIdx).wFluct;

        // get the tau values from the Eulerian grid for the particle value
        double current_txx = eul->interp3D(turb->txx,"tau");
        double current_txy = eul->interp3D(turb->txy,"tau");
        double current_txz = eul->interp3D(turb->txz,"tau");
        double current_tyy = eul->interp3D(turb->tyy,"tau");
        double current_tyz = eul->interp3D(turb->tyz,"tau");
        double current_tzz = eul->interp3D(turb->tzz,"tau");

        // set tau_old to the interpolated values for each position
        newParticles.at(pIdx).txx_old = current_txx;
        newParticles.at(pIdx).txy_old = current_txy;
        newParticles.at(pIdx).txz_old = current_txz;
        newParticles.at(pIdx).tyy_old = current_tyy;
        newParticles.at(pIdx).tyz_old = current_tyz;
        newParticles.at(pIdx).tzz_old = current_tzz;

        // set delta_velFluct values to zero for now
        newParticles.at(pIdx).delta_uFluct = 0.0;
        newParticles.at(pIdx).delta_vFluct = 0.0;
        newParticles.at(pIdx).delta_wFluct = 0.0;

        // set isRogue to false and isActive to true for each particle
        newParticles.at(pIdx).isRogue = false;
        newParticles.at(pIdx).isActive = true;
        
    }

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


    // make a variable to keep track of the number of particles. Make sure it is the most updated value
    int nPar = pointList.size();
    


    currentFile = debugOutputFolder + "/particle_txx_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).txx_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_txy_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).txy_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_txz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).txz_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tyy_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tyy_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tyz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tyz_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_tzz_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).tzz_old);
    }
    fclose(fzout);


    currentFile = debugOutputFolder + "/particle_uFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).uFluct_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_vFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).vFluct_old);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_wFluct_old.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).wFluct_old);
    }
    fclose(fzout);



    currentFile = debugOutputFolder + "/particle_uFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).uFluct);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_vFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).vFluct);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_wFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).wFluct);
    }
    fclose(fzout);


    currentFile = debugOutputFolder + "/particle_delta_uFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_uFluct);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_delta_vFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_vFluct);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_delta_wFluct.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).delta_wFluct);
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
        fprintf(fzout,"%lf\n",pointList.at(idx).xPos);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_yPos.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).yPos);
    }
    fclose(fzout);

    currentFile = debugOutputFolder + "/particle_zPos.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nPar; idx++)
    {
        fprintf(fzout,"%lf\n",pointList.at(idx).zPos);
    }
    fclose(fzout);


    // now that all is finished, clean up the file pointer
    fzout = NULL;

}
