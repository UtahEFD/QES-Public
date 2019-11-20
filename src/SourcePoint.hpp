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
    
    SourcePoint( const vec3 &pt, const int numParticles, ParticleReleaseType rType,
                        const double domainXstart, const double domainXend, 
                        const double domainYstart, const double domainYend,
                        const double domainZstart, const double domainZend )
        : SourceKind( numParticles, rType ),
          m_pt( pt )
    {
        if( pt.e11 < domainXstart || pt.e11 > domainXend )
        {
            std::cerr << "ERROR (SourcePoint::SourcePoint): pt.e11 is outside of domain! pt.e11 = \"" << pt.e11 
                << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"\n";
            exit(1);
        }
        if( pt.e21 < domainYstart || pt.e21 > domainYend )
        {
            std::cerr << "ERROR (SourcePoint::SourcePoint): pt.e11 is outside of domain! pt.e21 = \"" << pt.e21 
                << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"\n";
            exit(1);
        }
        if( pt.e31 < domainZstart || pt.e31 > domainZend )
        {
            std::cerr << "ERROR (SourcePoint::SourcePoint): pt.e11 is outside of domain! pt.e31 = \"" << pt.e31 
                << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"\n";
            exit(1);
        }
    }

    ~SourcePoint()
    {
    }

    virtual void parseValues()
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
