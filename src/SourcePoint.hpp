#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourcePoint : public SourceKind
{
private:
    vec3 m_pt;
    
protected:
    
public:

    // Default constructor
    SourcePoint()
    {
    }
    
    SourcePoint( const vec3 &pt, const int numParticles, ParticleReleaseType rType )
        : SourceKind( numParticles, rType ),
          m_pt( pt )
    {
    }

    ~SourcePoint()
    {
    }

    virtual void parseValue()
    {
        // Pete can help fill this in later, but
        // it would need to do the following:
        
        // <pointSource>
        // <numParticles>100000</numParticles>
        // <posX> 10.0 </posX>
        // <posY> 100.0 </posY>
        // <posZ> 50.0 </posZ>
        // </pointSource>  
    }
    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle> &emittedParticles);
    
};
