
#pragma once


#include "ReleaseType.hpp"


class ReleaseType_duration : public ReleaseType
{
private:

    // note that this also inherits data members ParticleReleaseType m_rType, int m_parPerTimestep, double m_releaseStartTime, 
    //  double m_releaseEndTime, and int m_numPar from ReleaseType.
    // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

    double releaseStartTime;
    double releaseEndTime;
    int parPerTimestep;
    
    
protected:
    
public:

    // Default constructor
    ReleaseType_duration()
    {
    }

    // destructor
    ~ReleaseType_duration()
    {
    }


    virtual void parseValues()
    {
        parReleaseType = ParticleReleaseType::duration;

        parsePrimitive<double>(true, releaseStartTime, "releaseStartTime");
        parsePrimitive<double>(true, releaseEndTime, "releaseEndTime");
        parsePrimitive<int>(true, parPerTimestep, "parPerTimestep");
        
    }


    void calcReleaseInfo(const double& timestep, const double& simDur)
    {
        // set the overall releaseType variables from the variables found in this class
        m_parPerTimestep = parPerTimestep;
        m_releaseStartTime = releaseStartTime;
        m_releaseEndTime = releaseEndTime;
        double releaseDur = releaseEndTime - releaseStartTime;
        int releaseSteps = std::ceil(releaseDur/timestep);
        m_numPar = parPerTimestep*releaseSteps;
    }

    
};
