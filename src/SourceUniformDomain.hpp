#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourceUniformDomain : public SourceKind
{
private:
    float m_rangeX, m_rangeY, m_rangeZ;
    float m_minX, m_minY, m_minZ;
    
protected:
    
public:

    // Default constructor
    SourceUniformDomain()
    {
    }
    
    SourceUniformDomain( float minX, float minY, float minZ,
                         float maxX, float maxY, float maxZ, const int numParticles )
        : SourceKind( numParticles, ParticleReleaseType::instantaneous ),
          m_rangeX( maxX-minX ), m_rangeY( maxY-minY ), m_rangeZ( maxZ-minZ ),
          m_minX( minX ), m_minY( minY ), m_minZ( minZ )
    {
    }

    SourceUniformDomain( int nx, int ny, int nz, const int numParticles )
        : SourceKind( numParticles, ParticleReleaseType::instantaneous ),
          m_rangeX( nx ), m_rangeY( ny ), m_rangeZ( nz ),
          m_minX( 0 ), m_minY( 0 ), m_minZ( 0 )
    {
    }

    ~SourceUniformDomain()
    {
    }

    virtual void parseValues()
    {
        // Pete can help fill this in later, but
        // it would need to do the following:
    }
    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle> &emittedParticles);
    
};
