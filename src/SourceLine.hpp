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
    
    SourceLine( const vec3 &pt0, const vec3 &pt1, const int numParticles, ParticleReleaseType rType,
                        const double domainXstart, const double domainXend, 
                        const double domainYstart, const double domainYend,
                        const double domainZstart, const double domainZend )
        : SourceKind( numParticles, rType ),
          m_pt0( pt0 ), m_pt1( pt1 )
    {
        if( pt0.e11 < domainXstart || pt0.e11 > domainXend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt0.e11 is outside of domain! pt0.e11 = \"" << pt0.e11 
                << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"\n";
            exit(1);
        }
        if( pt0.e21 < domainYstart || pt0.e21 > domainYend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt0.e11 is outside of domain! pt0.e21 = \"" << pt0.e21 
                << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"\n";
            exit(1);
        }
        if( pt0.e31 < domainZstart || pt0.e31 > domainZend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt0.e11 is outside of domain! pt0.e31 = \"" << pt0.e31 
                << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"\n";
            exit(1);
        }

        if( pt1.e11 < domainXstart || pt1.e11 > domainXend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt1.e11 is outside of domain! pt1.e11 = \"" << pt1.e11 
                << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"\n";
            exit(1);
        }
        if( pt1.e21 < domainYstart || pt1.e21 > domainYend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt1.e11 is outside of domain! pt1.e21 = \"" << pt1.e21 
                << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"\n";
            exit(1);
        }
        if( pt1.e31 < domainZstart || pt1.e31 > domainZend )
        {
            std::cerr << "ERROR (SourceLine::SourceLine): pt1.e11 is outside of domain! pt1.e31 = \"" << pt1.e31 
                << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"\n";
            exit(1);
        }
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
