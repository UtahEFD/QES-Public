//
//  ReleaseType.hpp
//  
//  This class represents a generic particle release type. The idea is to make other classes that inherit from this class
//  that are the specific release types, that make it easy to set the desired particle information for a given release type
//
//  Created by Loren Atwood on 01/25/20.
//

#pragma once


#include <cmath>


#include "util/ParseInterface.h"


enum ParticleReleaseType {
    instantaneous,
    continuous,
    duration
};


class ReleaseType : public ParseInterface
{
protected:

    
public:

    // this is a description variable for determining the source release type. May or may not be used.
    // !!! this needs set by parseValues() in each source generated from input files.
    ParticleReleaseType parReleaseType;
    
    // LA-future work: might need to add another variable for the total number of available particles,
    //  or have a checking function that compares numParticles with totalNumParticles.

    int m_parPerTimestep;   // this is the number of particles a given source needs to release each timestep
    double m_releaseStartTime;  // this is the time a given source should start releasing particles
    double m_releaseEndTime;    // this is the time a given source should end releasing particles
    int m_numPar;   // this is the total number of particles expected to be released by a given source over the entire simulation
    
    
    // default constructor
    ReleaseType()
    {
    }
    
    // destructor
    virtual ~ReleaseType()
    {
    }



    // this function is used to parse all the variables for each release type in a given source from the input .xml file
    // each release type overloads this function with their own version, allowing different combinations of input variables for each release type, 
    // all these differences handled by parseInterface().
    // The = 0 at the end should force each inheriting class to require their own version of this function
    // !!! in order for all the different combinations of input variables to work properly for each source, this function requires 
    //  manually setting the variable parReleaseType in each version found in release types that inherit from this class.
    //  This is in addition to any other variables required for an individual release type that inherits from this class.
    virtual void parseValues() = 0;


    // this function is for setting the required inherited variables int m_parPerTimestep, double m_releaseStartTime, 
    //  double m_releaseEndTime, and m_numPar. The way this is done differs for each release type inheriting from this class.
    // Note that this is a pure virtual function - enforces that the derived class MUST define this function
    //  this is done by the = 0 at the end of the function.
    // !!! Care must be taken to set all these variables in each inherited version of this function, where the calculated values
    //  will be able to pass the call to checkReleaseInfo() by the class setting up a vector of all the sources.
    // !!! each release type needs to have this function manually called for them by whatever class sets up a vector of this class.
    // LA-future work: The specific way time is handled still needs to be worked out to stay consistent throughout the entire program.
    virtual void calcReleaseInfo(const double& timestep, const double& simDur) = 0;


    // this function is for checking the set release type variables to make sure they are consistent with simulation information.
    // !!! each release type needs to have this function manually called for them by whatever class sets up a vector of this class.
    // LA-note: the check functions are starting to be more diverse and in different spots.
    //  Maybe a better name for this function would be something like checkReleaseTypeInfo().
    // LA-warn: should this be virtual? The idea is that I want it to stay as this function no matter what ReleaseType is chosen,
    //  I don't want this function overloaded by any classes inheriting this class.
    // LA-future work: the methods for calculating the times is off, I see one too many timesteps in some of my output plots.
    //  so this means that some of the methods for checking the number of particles needs to change cause they depend on a time calculation that 
    //  needs to change.
    virtual void checkReleaseInfo(const double& timestep, const double& simDur)
    {
        if( m_parPerTimestep <= 0 )
        {
            std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_parPerTimestep is <= 0!";
            std::cerr << " m_parPerTimestep = \"" << m_parPerTimestep << "\"" << std::endl;
            exit(1);
        }
        if( m_releaseStartTime < 0 )
        {
            std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseStartTime is < 0!";
            std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\"" << std::endl;
            exit(1);
        }
        if( m_releaseEndTime > simDur )
        {
            std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseEndTime is > input simDur!";
            std::cerr << " m_releaseEndTime = \"" << m_releaseEndTime << "\", simDur = \"" << simDur << "\"" << std::endl;
            exit(1);
        }
        if( m_releaseEndTime < m_releaseStartTime )
        {
            std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseEndTime is < input m_releaseStartTime!";
            std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\", m_releaseEndTime = \"" << m_releaseEndTime << "\"" << std::endl;
            exit(1);
        }

        // this one is a bit trickier to check. Specifically the way the number of timesteps for a given release 
        //  is calculated needs to be watched carefully to make sure it is consistent throughout the entire program
        double releaseDur = m_releaseEndTime - m_releaseStartTime;
        if( parReleaseType == ParticleReleaseType::instantaneous )
        {
            if( releaseDur != 0)
            {
                std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is instantaneous but input m_releaseStartTime does not equal m_releaseEndTime!";
                std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\", m_releaseEndTime = \"" << m_releaseEndTime << "\"" << std::endl;
                exit(1);
            }
            if( m_numPar != m_parPerTimestep )
            {
                std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is instantaneous but input m_numPar does not equal input m_parPerTimestep!";
                std::cerr << " m_numPar = \"" << m_numPar << "\", m_parPerTimestep = \"" << m_parPerTimestep << "\"" << std::endl;
                exit(1);
            }
        } else
        {
            // Again, the way the number of timesteps for a given release 
            //  is calculated needs to be watched carefully to make sure it is consistent throughout the program
            int releaseSteps = std::ceil(releaseDur/timestep);
            if( releaseSteps == 0 )
            {
                std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is not instantaneous but calculated releaseSteps is zero!";
                std::cerr << " releaseDur = \"" << releaseDur << "\", timestep = \"" << timestep << "\"" << std::endl;
                exit(1);
            }
            if( m_parPerTimestep*releaseSteps != m_numPar )
            {
                std::cerr << "ERROR (ReleaseType::checkReleaseInfo): calculated particles for release does not match input m_numPar!";
                std::cerr << " m_parPerTimestep = \"" << m_parPerTimestep << "\", releaseSteps = \"" << releaseSteps 
                        << "\", m_numPar = \"" << m_numPar << "\"" << std::endl;
                exit(1);
            }
        }

    }



};
