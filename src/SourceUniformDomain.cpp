#include <random>
#include "SourceUniformDomain.hpp"

int SourceUniformDomain::emitParticles(const float dt,
                               const float currTime,
                               std::vector<particle> &emittedParticles)
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

            for (int pidx=0; pidx<m_totalParticles; pidx++) {

                particle cPar;

                // generate uniform dist in domain
                cPar.pos.e11 = uniformDistr(prng) * m_rangeX + m_minX;
                cPar.pos.e21 = uniformDistr(prng) * m_rangeY + m_minY;
                cPar.pos.e31 = uniformDistr(prng) * m_rangeZ + m_minZ;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    }

    return emittedParticles.size();
    
}
