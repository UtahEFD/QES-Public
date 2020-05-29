
#pragma once


#include "SourceKind.hpp"


class SourceFullDomain : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    // this source is a bit weird because the domain size has to be obtained after the input parser.
    //  this would mean either doing a function call unique to this source to supply the required data during the dispersion constructor
    //  or by using checkPosInfo() differently than it is normally intended to set the domain size variables
    double xDomainStart;
    double yDomainStart;
    double zDomainStart;
    double xDomainEnd;
    double yDomainEnd;
    double zDomainEnd;
    
protected:
    
public:

    // Default constructor
    SourceFullDomain()
    {
    }

    // destructor
    ~SourceFullDomain()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::fullDomain;

        setReleaseType();

    }


    void checkPosInfo( const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt, const float currTime,
                      std::list<Particle>& emittedParticles);
    
};
