#include "SourcePoint.hpp"


void SourcePoint::checkMetaData( const double& domainXstart, const double& domainXend, 
                                 const double& domainYstart, const double& domainYend,
                                 const double& domainZstart, const double& domainZend)
{
    // not even sure how to do checks on the ParticleReleaseType. Maybe call a function to check it that is inherited from the overall class that defines it?
    //if ( m_rType )

    if( m_numParticles <= 0 )
    {
        std::cerr << "ERROR (SourcePoint::checkMetaData): input numParticles is <= 0! numParticles = \"" << m_numParticles << "\"" << std::endl;
        exit(1);
    }

    if( posX < domainXstart || posX > domainXend )
    {
        std::cerr << "ERROR (SourcePoint::checkMetaData): input posX is outside of domain! posX = \"" << posX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( posY < domainYstart || posY > domainYend )
    {
        std::cerr << "ERROR (SourcePoint::checkMetaData): input posY is outside of domain! posY = \"" << posY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( posZ < domainZstart || posZ > domainZend )
    {
        std::cerr << "ERROR (SourcePoint::checkMetaData): input posZ is outside of domain! posZ = \"" << posZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourcePoint::emitParticles( const float dt,
                                const float currTime,
                                std::vector<particle>& emittedParticles)
{
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
        
    } else if( m_rType == ParticleReleaseType::perTimeStep )
    {

        //int parPerTimestep = m_numParticles*dt/simDur;
        int parPerTimestep = 10;  // need to fix this, but requires changes to all sources to get the simDur variable somehow
                                  // 100,000*0.1/1,000 = 10

        for (int pidx = 0; pidx < parPerTimestep; pidx++) {  // this might also need fixed

            particle cPar;
            cPar.xPos = posX;
            cPar.yPos = posY;
            cPar.zPos = posZ;

            cPar.tStrt = currTime;
            
            emittedParticles.push_back( cPar );
        }

    } else
    {
        std::cerr << "ERROR (SourcePoint::emitParticles): ParticleReleaseType \"" << m_rType << "\" has not been implemented in code yet!" << std::endl;
    }

    return emittedParticles.size();
    
}
