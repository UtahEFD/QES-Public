//
// Created by Fabien Margairaz on 10/3/23.
//

#include "SourceFire.h"

int SourceFire::emitParticles(const float &dt,
                              const float &currTime,
                              std::list<Particle *> &emittedParticles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (active) {

    for (int pidx = 0; pidx < particle_per_time; pidx++) {

      // Particle *cPar = new Particle();
      Particle *cPar = new ParticleTracer();
      cPar->xPos_init = x;
      cPar->yPos_init = y;
      cPar->zPos_init = z;

      cPar->tStrt = currTime;
      cPar->sourceIdx = sourceIdx;
      emittedParticles.push_front(cPar);
    }
  }

  return (int)emittedParticles.size();
}