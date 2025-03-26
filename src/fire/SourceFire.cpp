/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file SourceFire.cpp
 * @brief This class specifies Fire sources for QES-Fire and QES-Plume integration
 */

#include "SourceFire.h"

void FireSourceBuilder::setSourceParam(const vec3& x,const QEStime& start,const QEStime& end, const int&ppt)
{
  m_x = x;
  m_start_time = start;
  m_end_time = end;
  m_ppt = ppt;
}

Source *FireSourceBuilder::create(QESDataTransport &)
{
  return nullptr;
}

Source* FireSourceBuilder::create()
{

  auto* source = new Source(new SourceReleaseController_base(m_start_time,m_end_time,m_ppt,0.0));
  source->addComponent(new SourceGeometryPoint(m_x));
  // source->addComponent(new SetPhysicalProperties(d, rho));
  // source->addComponent(...);
  return source;
}

/*int SourceFire::emitParticles(const float &dt,
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

  return 0;//(int)emittedParticles.size();
  }*/
