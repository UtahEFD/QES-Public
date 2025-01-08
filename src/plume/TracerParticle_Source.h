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

/** @file Source__TracerParticles.h
 * @brief  This class represents a generic source type
 */

#pragma once

#include <random>
#include <list>

#include "ManagedContainer.h"
#include "Source.h"
#include "TracerParticle.h"


#include "PI_Source.hpp"
#include "PI_SourceGeometry.hpp"
#include "PI_SourceGeometry_Cube.hpp"
#include "PI_SourceGeometry_FullDomain.hpp"
#include "PI_SourceGeometry_Line.hpp"
#include "PI_SourceGeometry_Point.hpp"
#include "PI_SourceGeometry_SphereShell.hpp"

#include "PI_ReleaseType.hpp"
#include "PI_ReleaseType_instantaneous.hpp"
#include "PI_ReleaseType_continuous.hpp"
#include "PI_ReleaseType_duration.hpp"

#include "Particle.h"
#include "ParticleIDGen.h"

#include "SourceComponent.h"

class Source_old
{
private:
protected:
  Source_old() = default;

  PI_SourceGeometry *m_sourceGeometry{};
  PI_ReleaseType *m_releaseType{};

  ParticleIDGen *id_gen = nullptr;

public:
  int sourceIdx = -1;

  // constructor
  Source_old(const int &sidx, const PI_Source *in)
  {
    sourceIdx = sidx;

    m_sourceGeometry = in->m_sourceGeometry;
    m_releaseType = in->m_releaseType;
    id_gen = ParticleIDGen::getInstance();
  }

  // destructor
  virtual ~Source_old() = default;

  virtual int getNewParticleNumber(const float &dt,
                                   const float &currTime)
  {
    if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
      return m_releaseType->m_particlePerTimestep;
    } else {
      return 0;
    }
  }
};

class TracerParticle_Source : public Source_old
{
public:
  TracerParticle_Source(const int &sidx, const PI_Source *in) : Source_old(sidx, in) {}
  // destructor
  ~TracerParticle_Source() override = default;

  int getNewParticleNumber(const float &dt,
                           const float &currTime) override;

  virtual void emitParticles(const float &dt,
                             const float &currTime,
                             ManagedContainer<TracerParticle> *particles);
};
