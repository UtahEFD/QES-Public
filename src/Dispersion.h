//
//  Dispersion.h
//  
//  This class handles dispersion information
//  So it contains and manages the overall particle list and overall source list
//  Plume updates the particle list during the simulation using the dispersion information
//

#ifndef DISPERSION_H
#define DISPERSION_H

#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <helper_math.h>


#include "Random.h"

#include "particle.hpp"

#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

#include "PlumeInputData.hpp"
#include "Eulerian.h"


class Dispersion {
    
    public:
        

        // constructor
        // starts by determining the domain size from the urb and turb grid information
        // then the sources created from the input .xml files by parse interface are taken and added to the list of sources
        // finally some of the important overall metadata for the set of particles is set to initial values
        //  and all metadata is calculated and checked for each source.
        //  At the same time as calculating and checking the metadata for each source, the total number of particles to release is calculated.
        Dispersion( PlumeInputData* PID,Urb* urb,Turb* turb,Eulerian* eul, const bool& debug_val);



        // these are not technically the same as the input urb and turb grid start and end variables
        // the domain size variables are determined from the input urb and turb grid start and end values
        //  using the determineDomainSize() function.
        double domainXstart;    // the domain starting x value found by the determineDomainSize() function
        double domainXend;      // the domain ending x value found by the determineDomainSize() function
        double domainYstart;    // the domain starting y value found by the determineDomainSize() function
        double domainYend;      // the domain ending y value found by the determineDomainSize() function
        double domainZstart;    // the domain starting z value found by the determineDomainSize() function
        double domainZend;      // the domain ending z value found by the determineDomainSize() function

        
        
        // 
        // This the storage for all particles
        // 
        // the sources can set these values, then the other values are set using urb and turb info using these values
        std::vector<particle> pointList;


        // ALL Sources that will be used 
        std::vector< SourceKind* > allSources;


        // this is the total number of particles expected to be released during the simulation
        // !!! this has to be calculated carefully inside the getInputSources() function
        int totalParsToRelease;


        // some overall metadata for the set of particles
        double isRogueCount;        // just a total number of rogue particles per time iteration
        double isNotActiveCount;       // just a total number of inactive active particles per time iteration
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no


        void setParticleVals(Turb* turb, Eulerian* eul, std::vector<particle>& newParticles);

        
    private:
    

        // get the domain size from the input urb and turb grids
        void determineDomainSize(Urb* urb, Turb* turb);

        // this function gets sources from input data and adds them to the allSources vector
        // this function also calls the many check and calc functions for all the input sources
        // !!! note that these check and calc functions have to be called here 
        //  because each source requires extra data not found in the individual source data
        // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
        void getInputSources(PlumeInputData* PID);


        // function for finding the largest sig value
        double getMaxVariance(const std::vector<double>& sigma_x_vals,const std::vector<double>& sigma_y_vals,const std::vector<double>& sigma_z_vals);


        // copies of debug related information from the input arguments
        bool debug;
        
};
#endif
