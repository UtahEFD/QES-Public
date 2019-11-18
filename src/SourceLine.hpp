#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourceLine : public SourceKind
{
private:
    vec3 m_pt0;
    vec3 m_pt1;
    
protected:
    
public:

    // Default constructor
    SourceLine()
    {
    }
    
    SourceLine( const vec3 &pt0, const vec3 &pt1, const int numParticles, ParticleReleaseType rType )
        : SourceKind( numParticles, rType ),
          m_pt0( pt0 ), m_pt1( pt1 )
    {
    }

    ~SourceLine()
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
