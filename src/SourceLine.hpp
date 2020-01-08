#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourceLine : public SourceKind
{
private:

    // note that this also inherits data members int m_numParticles and ParticleReleaseType m_rType from SourceKind.
    //  this also inherits data members SourceShape m_sShape and std::string inputReleaseType from SourceKind.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double posX_0;
    double posY_0;
    double posZ_0;
    double posX_1;
    double posY_1;
    double posZ_1;
    
protected:
    
public:

    // Default constructor
    SourceLine()
    {
    }
    
    // specialized constructor with initializer list
    SourceLine( const double& xPos0, const double& yPos0, const double& zPos0,
                const double& xPos1, const double& yPos1, const double& zPos1,
                const int& numParticles, const ParticleReleaseType& rType,
                const double& domainXstart, const double& domainXend, 
                const double& domainYstart, const double& domainYend,
                const double& domainZstart, const double& domainZend )
        : SourceKind( numParticles, rType ),
          posX_0( xPos0 ), posY_0( yPos0 ), posZ_0( zPos0 ),
          posX_1( xPos1 ), posY_1( yPos1 ), posZ_1( zPos1 )
    {
        m_sShape = SourceShape::line;

        checkMetaData(domainXstart,domainXend,domainYstart,domainYend,domainZstart,domainZend);
    }

    // destructor
    ~SourceLine()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::line;

        parsePrimitive<std::string>(true, inputReleaseType, "releaseType");
        parsePrimitive<int>(true, m_numParticles, "numParticles");
        
        parsePrimitive<double>(true, posX_0, "posX_0");
        parsePrimitive<double>(true, posY_0, "posY_0");
        parsePrimitive<double>(true, posZ_0, "posZ_0");
        parsePrimitive<double>(true, posX_1, "posX_1");
        parsePrimitive<double>(true, posY_1, "posY_1");
        parsePrimitive<double>(true, posZ_1, "posZ_1");

        setReleaseType(inputReleaseType);
    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
