//
//  Dispersion.h
//  
//  This class handles dispersion information
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
        
        Dispersion(Urb*,Turb*,PlumeInputData*,Eulerian*,const std::string& debugOutputFolder_val); // starts by determining the domain size from the urb and turb grid information
                                                                                        // then there are some lines that can be activated by changing the compiler flag to 1 to manually add 
                                                                                        //  some source directly from specialized constructors for debugging
                                                                                        // then the sources created from the input .xml files by parse interface are taken and added to the list of sources
                                                                                        // finally some of the important overall metadata for the set of particles is set to initial values

                                                
        // looks like this is just defining a list of information for Plume to use to know where and when to release different particles.
        // so defining where and when every single particle is released over the entire simulation.
        // this also holds the full list of particle information, so the dispersion class could probably be renamed to Lagrangian since it is the Lagrangian grid of values
        

        // yup, need the domain size in this, for the checkMetaData() function
        // so need to write a function that figures out the domain size from the urb and turb grids
        // maybe for now just use the preexisting turb grid just to be safe
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


        // some overall metadata for the set of particles
        double isRogueCount;        // just a total number of rogue particles per time iteration
        double isActiveCount;       // just a total number of active particles per time iteration
        double vel_threshold;       // the velocity fluctuation threshold velocity used to determine if particles are rogue or no


        void setParticleVals(Turb* turb, Eulerian* eul, std::vector<particle>& newParticles);

        
        // this is the output folder for debug variable output
        std::string debugOutputFolder;
        
        void outputVarInfo_text();


    private:
    

        // get the domain size from the input urb and turb grids
        void determineDomainSize(Urb* urb, Turb* turb);

        // this function takes the sources from PlumeInputData and puts them into the allSources vector found in dispersion
        // this also calls the check metadata function for the input sources before adding them to the list.
        // the check metadata function should already have been called for all the other sources during the specialized constructor phases used to create them.
        void getInputSources(PlumeInputData* PID);


        // function for finding the largest sig value, which could be used for other similar datatypes if needed
        double getMaxVariance(const std::vector<double>& sigma_x_vals,const std::vector<double>& sigma_y_vals,const std::vector<double>& sigma_z_vals);

};
#endif
