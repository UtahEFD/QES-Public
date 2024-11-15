/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/** @file Source_TracerParticles.cpp
 * @brief  This class represents a generic source type
 */

#include "TracerParticle_Source.h"

int TracerParticle_Source::getNewParticleNumber(const float &dt, const float &currTime)
{
  if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
    return m_releaseType->m_particlePerTimestep;
  } else {
    return 0;
  }
}

void TracerParticle_Source::emitParticles(const float &dt,
                                          const float &currTime,
                                          ManagedContainer<TracerParticle> *particles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
    if (!particles->check_size(m_releaseType->m_particlePerTimestep)) {
      std::cerr << "[ERROR] particle container ill formed (not enough space)" << std::endl;
      exit(1);
    }
    for (int pidx = 0; pidx < m_releaseType->m_particlePerTimestep; ++pidx) {
      // Particle *cPar = m_particleTypeFactory->Create(m_protoParticle);
      // m_protoParticle->setParticleParameters(cPar);
      particles->insert();
      m_sourceGeometry->setInitialPosition(particles->last_added()->pos_init);
      // m_protoParticle->setParticleParameters(particles->last_added());
      particles->last_added()->tStrt = currTime;
      particles->last_added()->sourceIdx = sourceIdx;
      particles->last_added()->m = m_releaseType->m_massPerParticle;
    }
    // int emitted = (int)particles->get_nbr_added();
  }
}
