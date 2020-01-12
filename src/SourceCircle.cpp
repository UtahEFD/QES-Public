#include "SourceCircle.hpp"


void SourceCircle::checkMetaData( const double& domainXstart, const double& domainXend, 
                                  const double& domainYstart, const double& domainYend,
                                  const double& domainZstart, const double& domainZend)
{
    // not even sure how to do checks on the ParticleReleaseType. Maybe call a function to check it that is inherited from the overall class that defines it?
    //if ( m_rType )

    if( m_numParticles <= 0 )
    {
        std::cerr << "ERROR (SourceCircle::checkMetaData): input numParticles is <= 0! numParticles = \"" << m_numParticles << "\"" << std::endl;
        exit(1);
    }

    if( radius < 0 )
    {
        std::cerr << "ERROR (SourceCircle::checkMetaData): input radius is negative! radius = \"" << radius << "\"" << std::endl;
        exit(1);
    }

    if( (posX-radius) < domainXstart || (posX+radius) > domainXend )
    {
        std::cerr << "ERROR (SourceCircle::checkMetaData): input posX+radius is outside of domain! posX = \"" << posX << "\" radius = \"" << radius
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( (posY-radius) < domainYstart || (posY+radius) > domainYend )
    {
        std::cerr << "ERROR (SourceCircle::checkMetaData): input posY+radius is outside of domain! posY = \"" << posY << "\" radius = \"" << radius
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( (posZ-radius) < domainZstart || (posZ+radius) > domainZend )
    {
        std::cerr << "ERROR (SourceCircle::checkMetaData): input posZ is outside of domain! posZ = \"" << posZ << "\" radius = \"" << radius
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourceCircle::emitParticles( const float dt,
                                 const float currTime,
                                 std::vector<particle>& emittedParticles)
{

    // note that this is not complete yet, it is acting like a point till we add in stuff corresponding to the radius
    if (m_rType == ParticleReleaseType::instantaneous) {

        // release all particles only if currTime is 0
        if (currTime < dt) {

            for (int pidx = 0; pidx < m_numParticles; pidx++) {

                particle cPar;
                cPar.xPos = posX;
                cPar.yPos = posY;
                cPar.zPos = posZ;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    } else
    {
        std::cerr << "ERROR (SourceCircle::emitParticles): ParticleReleaseType \"" << m_rType << "\" has not been implemented in code yet!" << std::endl;
    }

    return emittedParticles.size();
    
}
