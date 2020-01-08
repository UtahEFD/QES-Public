#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourceCube : public SourceKind
{
private:

    // note that this also inherits data members int m_numParticles and ParticleReleaseType m_rType from SourceKind.
    //  this also inherits data members SourceShape m_sShape and std::string inputReleaseType from SourceKind.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double m_minX;
    double m_minY;
    double m_minZ;
    double m_maxX;
    double m_maxY;
    double m_maxZ;
    
protected:
    
public:

    // Default constructor
    SourceCube()
    {
    }
    
    // specialized constructor with initializer list
    SourceCube( const double& minX, const double& minY, const double& minZ,
                const double& maxX, const double& maxY, const double& maxZ,
                const int& numParticles, const ParticleReleaseType& rType,
                const double& domainXstart, const double& domainXend, 
                const double& domainYstart, const double& domainYend,
                const double& domainZstart, const double& domainZend )
        : SourceKind( numParticles, rType ),
          m_minX( minX ), m_minY( minY ), m_minZ( minZ ),
          m_maxX( maxX-minX ), m_maxY( maxY-minY ), m_maxZ( maxZ-minZ )
    {
        // notice that the ParticleReleaseType m_rType is set by the initializer list to always be instantaneous

        m_sShape = SourceShape::cube;

        checkMetaData(domainXstart,domainXend,domainYstart,domainYend,domainZstart,domainZend);
    }

    // destructor
    ~SourceCube()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::cube;
        
        parsePrimitive<std::string>(true, inputReleaseType, "releaseType");
        parsePrimitive<int>(true, m_numParticles, "numParticles");

        parsePrimitive<double>(true, m_minX, "minX");
        parsePrimitive<double>(true, m_minY, "minY");
        parsePrimitive<double>(true, m_minZ, "minZ");
        parsePrimitive<double>(true, m_maxX, "maxX");
        parsePrimitive<double>(true, m_maxY, "maxY");
        parsePrimitive<double>(true, m_maxZ, "maxZ");

        setReleaseType(inputReleaseType);
    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
