#pragma once


#include "SourceKind.hpp"


class SourcePoint : public SourceKind
{
private:

    // note that this also inherits data members int m_numParticles and ParticleReleaseType m_rType from SourceKind.
    //  this also inherits data members SourceShape m_sShape and std::string inputReleaseType from SourceKind.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double posX;
    double posY;
    double posZ;
    
protected:
    
public:

    // Default constructor
    SourcePoint()
    {
    }
    
    // specialized constructor with initializer list
    SourcePoint( const double& xPos, const double& yPos, const double& zPos, 
                 const int& numParticles, const ParticleReleaseType& rType,
                 const double& domainXstart, const double& domainXend, 
                 const double& domainYstart, const double& domainYend,
                 const double& domainZstart, const double& domainZend )
        : SourceKind( numParticles, rType ),
          posX( xPos ), posY( yPos), posZ( zPos )
    {
        m_sShape = SourceShape::point;

        checkMetaData(domainXstart,domainXend,domainYstart,domainYend,domainZstart,domainZend);
    }

    // destructor
    ~SourcePoint()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::point;

        parsePrimitive<std::string>(true, inputReleaseType, "releaseType");
        parsePrimitive<int>(true, m_numParticles, "numParticles");

        parsePrimitive<double>(true, posX, "posX");
        parsePrimitive<double>(true, posY, "posY");
        parsePrimitive<double>(true, posZ, "posZ");

        setReleaseType(inputReleaseType);
    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);
    
    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
