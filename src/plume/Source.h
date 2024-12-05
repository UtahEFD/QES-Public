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

/** @file SourceType.hpp
 * @brief  This class represents a generic sourece type
 *
 * @note Pure virtual child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include <random>
#include <list>

#include "util/ParseInterface.h"

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

class Source
{
private:
protected:
  Source() = default;

  ParticleType m_pType{};
  SourceShape m_sGeom{};
  ParticleReleaseType m_rType{};

  // ParticleTypeFactory *m_particleTypeFactory{};
  ParseParticle *m_protoParticle{};
  PI_SourceGeometry *m_sourceGeometry{};
  PI_ReleaseType *m_releaseType{};

  ParticleIDGen *id_gen = nullptr;

public:
  // this is the index of the source in the dispersion class overall list of sources
  // this is used to set the source ID for a given particle, to know from which source each particle comes from
  // !!! this will only be set correctly if a call to setSourceIdx() is done by the class that sets up a vector of this class.
  int sourceIdx = -1;

  // accessor to particle type
  // ParticleType particleType()
  //{
  //  return m_pType;
  //}
  // accessor to geometry type
  SourceShape geometryType()
  {
    return m_sGeom;
  }
  // accessor to release type
  ParticleReleaseType releaseType()
  {
    return m_rType;
  }
  /*int getNumParticles()
  {
    return m_releaseType->m_numPar;
  }*/
  // this is a pointer to the release type, which is expected to be chosen by parseValues() by each source via a call to setReleaseType().
  // this data structure holds information like the total number of particles to be released by the source, the number of particles to release
  // per time for each source, and the start and end times to be releasing from the source.
  // !!! this needs set by parseValues() in each source generated from input files by a call to the setReleaseType() function

  // LA-future work: need a class similar to ReleaseType that describes the input source mass.
  //  This could be mass, mass per time, volume with a density, and volume per time with a density.


  // constructor
  Source(const int &sidx, const PI_Source *in)
  {
    sourceIdx = sidx;

    m_sourceGeometry = in->m_sourceGeometry;
    m_releaseType = in->m_releaseType;

    // set types
    m_sGeom = m_sourceGeometry->m_sGeom;
    m_rType = m_releaseType->parReleaseType;

    id_gen = ParticleIDGen::getInstance();
  }

  // destructor
  virtual ~Source() = default;

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
