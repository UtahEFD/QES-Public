#pragma once


#include "SourceKind.hpp"


class SourceFullDomain : public SourceKind
{
private:

    // note that this also inherits data members int m_numParticles and ParticleReleaseType m_rType from SourceKind.
    //  this also inherits data members SourceShape m_sShape and std::string inputReleaseType from SourceKind.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    // this source is a bit weird because the domain size has to be obtained after the input parser.
    //  this would mean either doing a function call unique to this source to supply the required data during the dispersion constructor
    //  or by using checkMetaData() differently than it is normally intended to set the domain size variables
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
    
    // specialized constructor with initializer list
    SourceFullDomain( const int& numParticles, const ParticleReleaseType& rType,
                      const double& domainXstart, const double& domainXend, 
                      const double& domainYstart, const double& domainYend,
                      const double& domainZstart, const double& domainZend )
        : SourceKind( numParticles, rType )
    {
        m_sShape = SourceShape::fullDomain;

        // notice that the initializer list does not set the domain start and domain end variables 
        //  like it is done in other source types, where the position range type variables are set in the initializer list
        //  instead, they will be set in checkMetaData(), which is not a standard use of checkMetaData() for most sources
        checkMetaData(domainXstart,domainXend,domainYstart,domainYend,domainZstart,domainZend);
    }

    // destructor
    ~SourceFullDomain()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::fullDomain;

        parsePrimitive<std::string>(true, inputReleaseType, "releaseType");
        parsePrimitive<int>(true, m_numParticles, "numParticles");

        setReleaseType(inputReleaseType);
    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
