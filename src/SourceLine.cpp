#include "SourceLine.hpp"


void SourceLine::checkMetaData( const double& domainXstart, const double& domainXend, 
                                const double& domainYstart, const double& domainYend,
                                const double& domainZstart, const double& domainZend)
{
    // not even sure how to do checks on the ParticleReleaseType. Maybe call a function to check it that is inherited from the overall class that defines it?
    //if ( m_rType )

    if( m_numParticles <= 0 )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input numParticles is <= 0! numParticles = \"" << m_numParticles << "\"" << std::endl;
        exit(1);
    }

    if( posX_0 < domainXstart || posX_0 > domainXend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posX_0 is outside of domain! posX_0 = \"" << posX_0 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( posY_0 < domainYstart || posY_0 > domainYend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posY_0 is outside of domain! posY_0 = \"" << posY_0 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( posZ_0 < domainZstart || posZ_0 > domainZend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posZ_0 is outside of domain! posZ_0 = \"" << posZ_0 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }

    if( posX_1 < domainXstart || posX_1 > domainXend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posX_1 is outside of domain! posX_1 = \"" << posX_1 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( posY_1 < domainYstart || posY_1 > domainYend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posY_1 is outside of domain! posY_1 = \"" << posY_1 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( posZ_1 < domainZstart || posZ_1 > domainZend )
    {
        std::cerr << "ERROR (SourceLine::checkMetaData): input posZ_1 is outside of domain! posZ_1 = \"" << posZ_1 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourceLine::emitParticles( const float dt,
                               const float currTime,
                               std::vector<particle>& emittedParticles)
{
    if (m_rType == ParticleReleaseType::instantaneous) {

        // release all particles only if currTime is 0
        if (currTime < dt) {

            for (int pidx = 0; pidx < m_numParticles; pidx++) {

                particle cPar;

                // generate random point on line between m_pt0 and m_pt1
                double diffX = posX_1 - posX_0;
                double diffY = posY_1 - posY_0;
                double diffZ = posZ_1 - posZ_0;

                float t = drand48();
                cPar.xPos = posX_0 + t * diffX;
                cPar.yPos = posY_0 + t * diffY;
                cPar.zPos = posZ_0 + t * diffZ;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    } else
    {
        std::cerr << "ERROR (SourceLine::emitParticles): ParticleReleaseType \"" << m_rType << "\" has not been implemented in code yet!" << std::endl;
    }

    return emittedParticles.size();
    
}
