#include "SourceLine.hpp"

int SourceLine::emitParticles(const float dt,
                               const float currTime,
                               std::vector<particle> &emittedParticles)
{
    if (m_rType == ParticleReleaseType::instantaneous) {

        // release all particles only if currTime is 0
        if (currTime < dt) {

            for (int pidx=0; pidx<m_totalParticles; pidx++) {

                particle cPar;

                // generate random point on line between m_pt0 and m_pt1
                vec3 diff;
                diff.e11 = m_pt1.e11 - m_pt0.e11;
                diff.e21 = m_pt1.e21 - m_pt0.e21;
                diff.e31 = m_pt1.e31 - m_pt0.e31;

                float t = drand48();
                cPar.pos.e11 = m_pt0.e11 + t * diff.e11;
                cPar.pos.e21 = m_pt0.e21 + t * diff.e21;
                cPar.pos.e31 = m_pt0.e31 + t * diff.e31;

                cPar.tStrt = currTime;
                
                emittedParticles.push_back( cPar );
            }

        }
        
    }

    return emittedParticles.size();
    
}
