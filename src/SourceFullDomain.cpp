#include <random>
#include "SourceFullDomain.hpp"


void SourceFullDomain::checkMetaData( const double& domainXstart, const double& domainXend, 
                                      const double& domainYstart, const double& domainYend,
                                      const double& domainZstart, const double& domainZend)
{
    // not even sure how to do checks on the ParticleReleaseType. Maybe call a function to check it that is inherited from the overall class that defines it?
    //if ( m_rType )

    if( m_numParticles <= 0 )
    {
        std::cerr << "ERROR (SourceFullDomain::checkMetaData): input numParticles is <= 0! numParticles = \"" << m_numParticles << "\"" << std::endl;
        exit(1);
    }

    // notice that setting the variables as I am doing right now is not the standard way of doing this function
    xDomainStart = domainXstart;
    yDomainStart = domainYstart;
    zDomainStart = domainZstart;
    xDomainEnd = domainXend;
    yDomainEnd = domainYend;
    zDomainEnd = domainZend;


    if( xDomainStart > xDomainEnd )
    {
        std::cerr << "ERROR (SourceFullDomain::checkMetaData): input xDomainStart is greater than input xDomainEnd! xDomainStart = \"" << xDomainStart 
            << "\" xDomainEnd = \"" << xDomainEnd << "\"" << std::endl;
        exit(1);
    }
    if( yDomainStart > yDomainEnd )
    {
        std::cerr << "ERROR (SourceFullDomain::checkMetaData): input yDomainStart is greater than input yDomainEnd! yDomainStart = \"" << yDomainStart 
            << "\" yDomainEnd = \"" << yDomainEnd << "\"" << std::endl;
        exit(1);
    }
    if( zDomainStart > zDomainEnd )
    {
        std::cerr << "ERROR (SourceFullDomain::checkMetaData): input zDomainStart is greater than input zDomainEnd! zDomainStart = \"" << zDomainStart 
            << "\" zDomainEnd = \"" << zDomainEnd << "\"" << std::endl;
        exit(1);
    }
    
    // unfortunately there is no easy way to check that the input domain sizes are correct, so the code could potentially fail later on
    //  cause there is no easy checking method to be implemented here
}


int SourceFullDomain::emitParticles( const float dt,
                                     const float currTime,
                                     std::vector<particle>& emittedParticles)
{
    // this function WILL fail if checkMetaData() is not called, because for once checkMetaData() acts to set the required data for using this function
    
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
                cPar.xPos = uniformDistr(prng)*(xDomainEnd-xDomainStart) + xDomainStart;
                cPar.yPos = uniformDistr(prng)*(yDomainEnd-yDomainStart) + yDomainStart;
                cPar.zPos = uniformDistr(prng)*(zDomainEnd-zDomainStart) + zDomainStart;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    } else
    {
        std::cerr << "ERROR (SourceFullDomain::emitParticles): ParticleReleaseType \"" << m_rType << "\" has not been implemented in code yet!" << std::endl;
    }

    return emittedParticles.size();
    
}
