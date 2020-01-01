#pragma once

#include "TypeDefs.hpp"
#include "SourceKind.hpp"

class SourceUniformDomain : public SourceKind
{
private:
    double m_rangeX, m_rangeY, m_rangeZ;
    double m_minX, m_minY, m_minZ;
    
protected:
    
public:

    // Default constructor
    SourceUniformDomain()
    {
    }
    
    SourceUniformDomain( const double& minX, const double& minY, const double& minZ,
                         const double& maxX, const double& maxY, const double& maxZ, const int& numParticles,
                         const double& domainXstart, const double& domainXend, 
                         const double& domainYstart, const double& domainYend,
                         const double& domainZstart, const double& domainZend )
        : SourceKind( numParticles, ParticleReleaseType::instantaneous ),
          m_rangeX( maxX-minX ), m_rangeY( maxY-minY ), m_rangeZ( maxZ-minZ ),
          m_minX( minX ), m_minY( minY ), m_minZ( minZ )
    {
        if( m_minX < domainXstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minX is less than domainXstart! m_minX = \"" << m_minX 
                << "\" domainXstart = \"" << domainXstart << "\"\n";
            exit(1);
        }
        if( m_minY < domainYstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minY is less than domainYstart! m_minY = \"" << m_minY 
                << "\" domainYstart = \"" << domainYstart << "\"\n";
            exit(1);
        }
        if( m_minZ < domainZstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minZ is less than domainZstart! m_minZ = \"" << m_minZ 
                << "\" domainZstart = \"" << domainZstart << "\"\n";
            exit(1);
        }
        if( m_rangeX > domainXend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeX is greater than domainXend! m_rangeX = \"" << m_rangeX 
                << "\" domainXend = \"" << domainXend << "\"\n";
            exit(1);
        }
        if( m_rangeY > domainYend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeY is greater than domainYend! m_rangeY = \"" << m_rangeY 
                << "\" domainYend = \"" << domainYend << "\"\n";
            exit(1);
        }
        if( m_rangeZ > domainZend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeZ is greater than domainZend! m_rangeZ = \"" << m_rangeZ 
                << "\" domainZend = \"" << domainZend << "\"\n";
            exit(1);
        }
    }

    SourceUniformDomain( const int& nx, const int& ny, const int& nz, const double& dx, const double& dy, const double& dz, const int& numParticles,
                         const double& domainXstart, const double& domainXend, 
                         const double& domainYstart, const double& domainYend,
                         const double& domainZstart, const double& domainZend,
                         const int fudge )  // for some odd reason adding in the dx, dy, and dz values causes it to overload to this function instead of the other
                                            // so I'm throwing in a fake unused argument to force it to not overload to this but to use the other version
                                            // I'm guessing it has to do with the arguments being the same number, and it is hard to distinguish between int, float, and double?
        : SourceKind( numParticles, ParticleReleaseType::instantaneous ),
          m_rangeX( nx*dx-domainXstart ), m_rangeY( ny*dy-domainYstart ), m_rangeZ( nz*dz-domainZstart ),
          m_minX( domainXstart ), m_minY( domainXstart ), m_minZ( domainXstart )
    {
        if( m_minX < domainXstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minX is less than domainXstart! m_minX = \"" << m_minX 
                << "\" domainXstart = \"" << domainXstart << "\"\n";
            exit(1);
        }
        if( m_minY < domainYstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minY is less than domainYstart! m_minY = \"" << m_minY 
                << "\" domainYstart = \"" << domainYstart << "\"\n";
            exit(1);
        }
        if( m_minZ < domainZstart )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_minZ is less than domainZstart! m_minZ = \"" << m_minZ 
                << "\" domainZstart = \"" << domainZstart << "\"\n";
            exit(1);
        }
        if( m_rangeX > domainXend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeX is greater than domainXend! m_rangeX = \"" << m_rangeX 
                << "\" domainXend = \"" << domainXend << "\"\n";
            exit(1);
        }
        if( m_rangeY > domainYend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeY is greater than domainYend! m_rangeY = \"" << m_rangeY 
                << "\" domainYend = \"" << domainYend << "\"\n";
            exit(1);
        }
        if( m_rangeZ > domainZend )
        {
            std::cerr << "ERROR (SourceUniformDomain::SourceUniformDomain): m_rangeZ is greater than domainZend! m_rangeZ = \"" << m_rangeZ 
                << "\" domainZend = \"" << domainZend << "\"\n";
            exit(1);
        }
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
