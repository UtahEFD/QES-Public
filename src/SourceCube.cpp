#include <random>
#include "SourceCube.hpp"


void SourceCube::checkMetaData( const double& domainXstart, const double& domainXend, 
                                const double& domainYstart, const double& domainYend,
                                const double& domainZstart, const double& domainZend)
{
    // not even sure how to do checks on the ParticleReleaseType. Maybe call a function to check it that is inherited from the overall class that defines it?
    //if ( m_rType )

    if( m_numParticles <= 0 )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input numParticles is <= 0! numParticles = \"" << m_numParticles << "\"" << std::endl;
        exit(1);
    }

    if( m_minX > m_maxX )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minX is greater than input maxX! minX = \"" << m_minX 
            << "\" maxX = \"" << m_maxX << "\"" << std::endl;
        exit(1);
    }
    if( m_minY > m_maxY )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minY is greater than input maxY! minY = \"" << m_minY 
            << "\" maxY = \"" << m_maxY << "\"" << std::endl;
        exit(1);
    }
    if( m_minZ > m_maxZ )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minZ is greater than input maxZ! minZ = \"" << m_minZ 
            << "\" maxZ = \"" << m_maxZ << "\"" << std::endl;
        exit(1);
    }

    if( m_minX < domainXstart )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minX is outside of domain! minX = \"" << m_minX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( m_minY < domainYstart )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minY is outside of domain! minY = \"" << m_minY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( m_minZ < domainZstart )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input minZ is outside of domain! minZ = \"" << m_minZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }

    if( m_maxX > domainXend )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input maxX is outside of domain! maxX = \"" << m_maxX 
            << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
        exit(1);
    }
    if( m_maxY > domainYend )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input maxY is outside of domain! maxY = \"" << m_maxY 
            << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
        exit(1);
    }
    if( m_maxZ > domainZend )
    {
        std::cerr << "ERROR (SourceCube::checkMetaData): input maxZ is outside of domain! maxZ = \"" << m_maxZ 
            << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
        exit(1);
    }
}


int SourceCube::emitParticles( const float dt,
                               const float currTime,
                               std::vector<particle>& emittedParticles)
{
    // 
    // Only do instantaneous release for this source
    //
    if (m_rType == ParticleReleaseType::instantaneous) {

        // release all particles only if currTime is 0
        if (currTime < dt) {

            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 prng(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution<> uniformDistr(0.0, 1.0);

            for (int pidx = 0; pidx < m_numParticles; pidx++) {

                particle cPar;

                // generate uniform dist in domain
                cPar.pos.e11 = uniformDistr(prng)*(m_maxX-m_minX) + m_minX;
                cPar.pos.e21 = uniformDistr(prng)*(m_maxY-m_minY) + m_minY;
                cPar.pos.e31 = uniformDistr(prng)*(m_maxZ-m_minZ) + m_minZ;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    } else
    {
        std::cerr << "ERROR (SourceCube::emitParticles): ParticleReleaseType \"" << m_rType << "\" has not been implemented in code yet!" << std::endl;
    }

    return emittedParticles.size();
    
}
