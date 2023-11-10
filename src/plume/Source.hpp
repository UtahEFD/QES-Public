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

#include "SourceGeometry.hpp"
#include "SourceGeometry_Cube.hpp"
#include "SourceGeometry_FullDomain.hpp"
#include "SourceGeometry_Line.hpp"
#include "SourceGeometry_Point.hpp"
#include "SourceGeometry_SphereShell.hpp"

#include "Particle.hpp"
#include "Particle_Tracer.hpp"
#include "Particle_Heavy.hpp"

#include "ParticleManager.h"
#include "ParticleContainers.h"
#include "ParticleFactories.hpp"

#include "ReleaseType.hpp"
#include "ReleaseType_instantaneous.hpp"
#include "ReleaseType_continuous.hpp"
#include "ReleaseType_duration.hpp"

// #include "Interp.h"
#include "util/ParseInterface.h"
#include "winds/WINDSGeneralData.h"

class Source;

class ParseSource : public ParseInterface
{
private:
protected:
  ParseParticle *m_protoParticle{};
  SourceGeometry *m_sourceGeometry{};
  ReleaseType *m_releaseType{};

public:
  // this is the index of the source in the dispersion class overall list of sources
  // this is used to set the source ID for a given particle, to know from which source each particle comes from
  // !!! this will only be set correctly if a call to setSourceIdx() is done by the class that sets up a vector of this class.
  int sourceIdx = -1;
  // Interp *interp;

  // this is a description variable for determining the source shape. May or may not be used.
  // !!! this needs set by parseValues() in each source generated from input files.
  // SourceShape m_sShape;

  // this is a pointer to the release type, which is expected to be chosen by parseValues() by each source via a call to setReleaseType().
  // this data structure holds information like the total number of particles to be released by the source, the number of particles to release
  // per time for each source, and the start and end times to be releasing from the source.
  // !!! this needs set by parseValues() in each source generated from input files by a call to the setReleaseType() function

  // LA-future work: need a class similar to ReleaseType that describes the input source mass.
  //  This could be mass, mass per time, volume with a density, and volume per time with a density.

  // LA-future work: need a class similar to ReleaseType that describes how to distribute the particles along the source geometry
  //  This could be uniform, random normal distribution, ... I still need to think of more distributions.
  // On second thought, this may not be possible to do like ReleaseType, it may need to be specific to each source geometry.
  //  so maybe it needs to be more like BoundaryConditions, where each source determines which pointer function to choose for emitting particles
  //  based on a string input describing the distribution. Really depends on how easy the implementation details become.


  // constructor
  ParseSource() = default;

  // destructor
  virtual ~ParseSource() = default;

  int getNumParticles()
  {
    return m_releaseType->m_numPar;
  }

  void setReleaseType();
  void setSourceGeometry();
  void setParticleType();

  ParticleType particleType()
  {
    return m_protoParticle->particleType;
  }
  // accessor to geometry type
  SourceShape geometryType()
  {
    return m_sourceGeometry->m_sGeom;
  }
  // accessor to release type
  ParticleReleaseType releaseType()
  {
    return m_releaseType->parReleaseType;
  }


  // this function is used to parse all the variables for each source from the input .xml file
  // each source overloads this function with their own version, allowing different combinations of input variables for each source,
  // all these differences handled by parseInterface().
  // The = 0 at the end should force each inheriting class to require their own version of this function
  // !!! in order for all the different combinations of input variables to work properly for each source, this function requires calls to the
  //  setReleaseType() function and manually setting the variable m_sShape in each version found in sources that inherit from this class.
  //  This is in addition to any other variables required for an individual source that inherits from this class.
  void parseValues() override
  {
    setReleaseType();
    setParticleType();
    setSourceGeometry();
  }

  // this function is for checking the source metadata to make sure all particles will be released within the domain.
  // There is one source so far (SourceFullDomain) that actually uses this function to set a few metaData variables
  //  specific to that source as well as to do checks to make sure particles stay within the domain. This is not a problem
  //  so long as it is done this way with future sources very sparingly.
  //  In other words, avoid using this function to set variables unless you have to.
  // !!! each source needs to have this function manually called for them by whatever class sets up a vector of this class.
  void checkPosInfo(const double &domainXstart,
                    const double &domainXend,
                    const double &domainYstart,
                    const double &domainYend,
                    const double &domainZstart,
                    const double &domainZend);

  void checkReleaseInfo(const double &timestep, const double &simDur);
  friend class Source;
};

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
  SourceGeometry *m_sourceGeometry{};
  ReleaseType *m_releaseType{};

public:
  // this is the index of the source in the dispersion class overall list of sources
  // this is used to set the source ID for a given particle, to know from which source each particle comes from
  // !!! this will only be set correctly if a call to setSourceIdx() is done by the class that sets up a vector of this class.
  int sourceIdx = -1;

  // accessor to particle type
  ParticleType particleType()
  {
    return m_pType;
  }
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
  int getNumParticles()
  {
    return m_releaseType->m_numPar;
  }
  // this is a pointer to the release type, which is expected to be chosen by parseValues() by each source via a call to setReleaseType().
  // this data structure holds information like the total number of particles to be released by the source, the number of particles to release
  // per time for each source, and the start and end times to be releasing from the source.
  // !!! this needs set by parseValues() in each source generated from input files by a call to the setReleaseType() function

  // LA-future work: need a class similar to ReleaseType that describes the input source mass.
  //  This could be mass, mass per time, volume with a density, and volume per time with a density.


  // constructor
  Source(const int &sidx, const ParseSource *in)
  {
    sourceIdx = sidx;

    m_protoParticle = in->m_protoParticle;
    m_sourceGeometry = in->m_sourceGeometry;
    m_releaseType = in->m_releaseType;

    // set types
    m_pType = m_protoParticle->particleType;
    m_sGeom = m_sourceGeometry->m_sGeom;
    m_rType = m_releaseType->parReleaseType;
  }

  // destructor
  virtual ~Source() = default;

  virtual int getNewParticleNumber(const float &dt,
                                   const float &currTime) = 0;

  virtual void emitParticles(const float &dt,
                             const float &currTime,
                             ParticleContainers *particles) = 0;
};
