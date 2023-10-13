/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file Source_Tracer.hpp
 * @brief  This class represents a generic source type
 */
#include "Source_TracerParticles.h"

int Source_Tracers::getNewParticleNumber(const float &dt, const float &currTime)
{
  if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
    return m_releaseType->m_parPerTimestep;
  } else {
    return 0;
  }
}

void Source_Tracers::emitParticles(const float &dt,
                                   const float &currTime,
                                   ParticleContainers *particles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
    if (!particles->tracer->check_size(m_releaseType->m_parPerTimestep)) {
      std::cerr << "[ERROR] particle container hill formed (not enough space)" << std::endl;
      exit(1);
    }
    for (int pidx = 0; pidx < m_releaseType->m_parPerTimestep; ++pidx) {
      // Particle *cPar = m_particleTypeFactory->Create(m_protoParticle);
      // m_protoParticle->setParticleParameters(cPar);
      particles->tracer->insert();
      m_sourceGeometry->setInitialPosition(particles->tracer->last_added()->xPos_init,
                                           particles->tracer->last_added()->yPos_init,
                                           particles->tracer->last_added()->zPos_init);
      particles->tracer->last_added()->tStrt = currTime;
      particles->tracer->last_added()->sourceIdx = sourceIdx;
    }
    // emitted = (int)m_particleList->nbr_added();
  }
}
