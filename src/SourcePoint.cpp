
#include "SourcePoint.hpp"


void SourcePoint::checkPosInfo( const double& domainXstart, const double& domainXend, 
                                const double& domainYstart, const double& domainYend,
                                const double& domainZstart, const double& domainZend)
{
    if( posX < domainXstart || posX > domainXend )
    {
        std::cerr << "ERROR (SourcePoint::checkPosInfo): input posX is outside of domain! posX = \"" << posX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( posY < domainYstart || posY > domainYend )
    {
        std::cerr << "ERROR (SourcePoint::checkPosInfo): input posY is outside of domain! posY = \"" << posY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( posZ < domainZstart || posZ > domainZend )
    {
        std::cerr << "ERROR (SourcePoint::checkPosInfo): input posZ is outside of domain! posZ = \"" << posZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourcePoint::emitParticles( const float dt,
                                const float currTime,
                                std::vector<particle>& emittedParticles)
{
    // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
    if( currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime )
    {

        for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

            particle cPar;
            cPar.xPos_init = posX;
            cPar.yPos_init = posY;
            cPar.zPos_init = posZ;

            cPar.tStrt = currTime;

            cPar.sourceIdx = sourceIdx;
            
            emittedParticles.push_back( cPar );
        }

    }

    return m_rType->m_parPerTimestep;
    
}
