
#include "SourceCube.hpp"


void SourceCube::checkPosInfo( const double& domainXstart, const double& domainXend, 
                               const double& domainYstart, const double& domainYend,
                               const double& domainZstart, const double& domainZend)
{
    if( m_minX > m_maxX )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minX is greater than input maxX! minX = \"" << m_minX 
            << "\" maxX = \"" << m_maxX << "\"" << std::endl;
        exit(1);
    }
    if( m_minY > m_maxY )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minY is greater than input maxY! minY = \"" << m_minY 
            << "\" maxY = \"" << m_maxY << "\"" << std::endl;
        exit(1);
    }
    if( m_minZ > m_maxZ )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minZ is greater than input maxZ! minZ = \"" << m_minZ 
            << "\" maxZ = \"" << m_maxZ << "\"" << std::endl;
        exit(1);
    }

    if( m_minX < domainXstart )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minX is outside of domain! minX = \"" << m_minX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( m_minY < domainYstart )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minY is outside of domain! minY = \"" << m_minY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( m_minZ < domainZstart )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input minZ is outside of domain! minZ = \"" << m_minZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }

    if( m_maxX > domainXend )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input maxX is outside of domain! maxX = \"" << m_maxX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( m_maxY > domainYend )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input maxY is outside of domain! maxY = \"" << m_maxY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( m_maxZ > domainZend )
    {
        std::cerr << "ERROR (SourceCube::checkPosInfo): input maxZ is outside of domain! maxZ = \"" << m_maxZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourceCube::emitParticles( const float dt, const float currTime,
                               std::list<Particle>& emittedParticles)
{
    // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
    if( currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime )
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 prng(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> uniformDistr(0.0, 1.0);

        for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

            Particle cPar;

            // generate uniform dist in domain
            cPar.xPos_init = uniformDistr(prng)*(m_maxX-m_minX) + m_minX;
            cPar.yPos_init = uniformDistr(prng)*(m_maxY-m_minY) + m_minY;
            cPar.zPos_init = uniformDistr(prng)*(m_maxZ-m_minZ) + m_minZ;

            cPar.tStrt = currTime;

            cPar.sourceIdx = sourceIdx;
            
            emittedParticles.push_front( cPar );
        }
    }

    return m_rType->m_parPerTimestep;
    
}
